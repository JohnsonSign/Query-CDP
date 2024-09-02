import os
import sys
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from pst_convolutions import PSTConv
from modules import pointnet2_utils
import pointnet2_utils
from transformer import *
import utils
from chamfer_distance import ChamferDistance


def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    # init weights/bias
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    # dims: (C_in,512,512,3)
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i-1], dims[i]))
        if i==len(dims)-1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class FoldingNetSingle(nn.Module):
    def __init__(self, dims):
        super(FoldingNetSingle, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, X):
        return self.mlp.forward(X)


class PointwiseMLP(nn.Sequential):
    '''Nxdin ->Nxd1->Nxd2->...-> Nxdout'''
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class ContrastiveLearningModel(nn.Module):
    def __init__(self, 
                radius=1.5, 
                nsamples=3*3, 
                representation_dim=1024, 
                num_classes=60,
                pretraining=True, 
                temperature=0.1):
        super(ContrastiveLearningModel, self).__init__()

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv_fc = nn.Linear(2048, representation_dim)

        self.token_dim = representation_dim
        self.emb_relu = False
        self.depth = 3
        self.heads = 8
        self.dim_head = 128
        self.mlp_dim = 2048

        self.pretraining = pretraining
        if self.pretraining:   
            self.temperature = temperature

            self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.token_dim))
            trunc_normal_(self.mask_token, std=.02)

            self.Fold1 = FoldingNetSingle((2*self.token_dim+3, 512, 512, 6)) # 3MLP
            self.Fold2 = FoldingNetSingle((2*self.token_dim+6, 512, 512, 6)) # 3MLP
            
            self.criterion_dist = ChamferDistance()
            self.criterion_global = torch.nn.CrossEntropyLoss()
            self.criterion_local = torch.nn.CrossEntropyLoss()
            

        else:

            self.fc_out0 = nn.Linear(2048, 512)
            self.fc_out_bn0 = nn.BatchNorm1d(512)
            self.fc_out_relu0 = nn.ReLU(inplace=True)
            self.fc_out1 = nn.Linear(512, num_classes)

    def forward(self, xyzs, epoch):

        device = xyzs.get_device()

        if self.pretraining:
            Batchsize, Sub_clips, L_sub_clip, N_point, C_xyz = xyzs.shape  # [B, S, L, N, 3] B: for one gpu
            clip2 = xyzs[:, -1, :, :, :] # last clip  B L N 3
            clip2 = torch.split(tensor=clip2, split_size_or_sections=1, dim=1) # L* [B,1,N,3]
            clip2_list = [torch.squeeze(input=clip2_frame, dim=1).contiguous() for clip2_frame in clip2] #  L* [B,N,3]

            clip_anchor = [] # B, L, N, 3
            clip1 = xyzs[:, -1, :, :, :]
            clip1 = torch.split(tensor=clip1, split_size_or_sections=1, dim=1) # L* [B,1,N,3]
            clip1_list = [torch.squeeze(input=clip1_frame, dim=1).contiguous() for clip1_frame in clip1] #  L* [B,N,3]
            for t, clip1_frame in enumerate(clip1_list):
                clip1_frame_idx = pointnet2_utils.furthest_point_sample(clip1_frame, N_point//4)
                clip1_frame_flipped_xyz = pointnet2_utils.gather_operation(clip1_frame.transpose(1,2).contiguous(), clip1_frame_idx)
                clip1_frame_xyz = clip1_frame_flipped_xyz.transpose(1,2).contiguous()
                clip_anchor.append(clip1_frame_xyz)

            clip2_xyz_rgb = []
            for t, clip2_frame in enumerate(clip2_list):
                clip2_frame_idx = pointnet2_utils.furthest_point_sample(clip2_frame, N_point)
                clip2_frame_flipped_xyz = pointnet2_utils.gather_operation(clip2_frame.transpose(1,2).contiguous(), clip2_frame_idx)
                clip2_frame_xyz = clip2_frame_flipped_xyz.transpose(1,2).contiguous()
                if t==0:
                    clip2_rgb = torch.tensor([0.5,0.47,0.58]).unsqueeze(0).unsqueeze(0).to(device)
                    clip2_rgb = clip2_rgb.expand(clip2_frame_xyz.size()[0], clip2_frame_xyz.size()[1], -1) 
                    clip2_frame_xyz = torch.cat((clip2_frame_xyz, clip2_rgb),dim=2)
                    clip2_xyz_rgb.append(clip2_frame_xyz)
                elif t==1:
                    clip2_rgb = torch.tensor([0.43,0.35,0.7]).unsqueeze(0).unsqueeze(0).to(device)
                    clip2_rgb = clip2_rgb.expand(clip2_frame_xyz.size()[0], clip2_frame_xyz.size()[1], -1) 
                    clip2_frame_xyz = torch.cat((clip2_frame_xyz, clip2_rgb),dim=2)
                    clip2_xyz_rgb.append(clip2_frame_xyz)
                elif t==2:
                    clip2_rgb = torch.tensor([0.32,0.24,0.81]).unsqueeze(0).unsqueeze(0).to(device)
                    clip2_rgb = clip2_rgb.expand(clip2_frame_xyz.size()[0], clip2_frame_xyz.size()[1], -1) 
                    clip2_frame_xyz = torch.cat((clip2_frame_xyz, clip2_rgb),dim=2)
                    clip2_xyz_rgb.append(clip2_frame_xyz)
                elif t==3:
                    clip2_rgb = torch.tensor([0.23,0.11,0.94]).unsqueeze(0).unsqueeze(0).to(device)
                    clip2_rgb = clip2_rgb.expand(clip2_frame_xyz.size()[0], clip2_frame_xyz.size()[1], -1) 
                    clip2_frame_xyz = torch.cat((clip2_frame_xyz, clip2_rgb),dim=2)
                    clip2_xyz_rgb.append(clip2_frame_xyz)
            clip2_xyz_rgb = torch.stack(clip2_xyz_rgb, dim=1).reshape(Batchsize, -1, C_xyz+3) # 6, 4, 256, 6

            xyzs = xyzs.reshape((-1, L_sub_clip, N_point, C_xyz)) # [B*S, L, N, 3]

        # backbone
        if not self.pretraining:
            Batchsize = xyzs.size()[0]
            xyzs = torch.split(xyzs,4,dim=1)   # 4*[B,L',N,3]
            NUM_CLIPS = len(xyzs)
            xyzs = torch.stack(xyzs,dim=1)     # [B,4,L',N,3]
            xyzs = xyzs.reshape((-1,xyzs.size()[2],xyzs.size()[3],xyzs.size()[4]))

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)  # (B*S, L, N, 3) (B*S, L, C, N)

        if self.pretraining:
            new_features = new_features.permute(0, 1, 3, 2)    # [B*S, L, N, C] 
            new_features = self.conv_fc(new_features)          # [B*S, L, N, C] 

            BS, L_out, N_out, C_out = new_features.shape
            assert(C_out==self.token_dim)

            new_xys = new_xys.reshape((Batchsize, Sub_clips, L_out, N_out, C_xyz))           # [B, S, L, N, 3]
            new_features = new_features.reshape((Batchsize, Sub_clips, L_out, N_out, C_out)) # [B, S, L, N, C]
            assert(L_out==1) 

            new_xys = torch.squeeze(new_xys, dim=2).contiguous()            # [B, S, N, 3]
            new_features = torch.squeeze(new_features, dim=2).contiguous()  # [B, S, N, C]

            label_global = torch.mean(input=new_features, dim=-2, keepdim=False)    # (B, S, C)
            label_global = torch.max(input=label_global, dim=1, keepdim=False)[0]   # (B, C)

            emb_xyzs, label_xyz = torch.split(new_xys, [Sub_clips-1, 1], dim=1)           # [B, S-1, N, 3], [B, 1, N, 3]
            emb_tokens, label_local = torch.split(new_features, [Sub_clips-1, 1], dim=1)  # [B, S-1, N, C], [B, 1, N, C]
            
            label_xyz = torch.squeeze(label_xyz, dim=1)       # [B, N, 3]
            label_local = torch.squeeze(label_local, dim=1)   # [B, N, C]

            negative_extra = emb_tokens[:,0:Sub_clips-2,:,:].contiguous() # B,S-2,N,C
            negative_extra = negative_extra.reshape(-1, self.token_dim) # B*(S-2)*N,C
            negative_extra = F.normalize(negative_extra, dim=-1)
            
            emb_tokens = emb_tokens.reshape((Batchsize, (Sub_clips-1)*N_out, C_out))      # [B, S-1*N, C]
            
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = Batchsize)
            emb_tokens = torch.cat((emb_tokens, cls_tokens), dim=1) # B, S-1*N + 1, C

            # xyzt pos embedding 
            xyzts = []
            xyz_list = torch.split(tensor=emb_xyzs, split_size_or_sections=1, dim=1)      # S-1*[B, 1, N, 3]
            xyz_list = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyz_list] # S-1*[B, N, 3]
            cls_token_T = len(xyz_list)+1
            for t, xyz in enumerate(xyz_list):
                # [B, N, 3]
                t = torch.ones((xyz.size()[0], xyz.size()[1], 1), dtype=torch.float32, device=device) * (t+1)
                xyzt = torch.cat(tensors=(xyz, t), dim=2) # [B, N, 4]
                xyzts.append(xyzt)
            xyzts = torch.stack(tensors=xyzts, dim=1) # [B, S-1, N, 4]

            xyzts_cls_token = torch.ones((xyzts.size()[0], 1), dtype=torch.float32, device=device)*cls_token_T
            xyzts_cls_xyz = torch.mean(input=label_xyz, dim=1, keepdim=False) # B,3
            xyzts_cls_token = torch.cat((xyzts_cls_xyz, xyzts_cls_token), dim=1).unsqueeze(1) # B, 1, 4


            xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))  # [B, S-1*N, 4]

            xyzts = torch.cat((xyzts, xyzts_cls_token), dim=1) # B, S-1*N+1, 4
            xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1) # [B, S-1*N+1, C]

            embedding = xyzts + emb_tokens  # [B, S-1*N+1, C]

            if self.emb_relu:
                embedding = self.emb_relu(embedding)

            # transformer
            output = self.transformer(embedding) # [B, S-1*N+1, C]
            output_global, _ = torch.split(output, [(Sub_clips-1)*N_out,1], dim=1) # B, S-1*N, C ; B, 1, C
            output = output.reshape(-1, output.size()[-1]) # B*((S-1)*N+1), C
            output = self.mlp_head(output)       # B*((S-1)*N+1), C

            output = output.reshape((Batchsize, -1, self.token_dim)) # B, S-1*N+1, C
            output, cls_token_global = torch.split(output, [(Sub_clips-1)*N_out,1], dim=1) # B, S-1*N, C ; B, 1, C
            cls_token_global = cls_token_global.squeeze(1) # B, C 
            output = output.reshape((Batchsize, Sub_clips-1, N_out, -1))
            mask_local = output[:,-1,:,:].contiguous()                     # [B, N, C]

            mask_local = F.softmax(self.network_pred(mask_local), dim=-1) # B, N, 2C
            mask_local = mask_local.permute(0,2,1) # B, 2C, N 
            mask_local = torch.einsum('bmn, mc->bcn',mask_local, self.mb) # B, C, N
            mask_local = mask_local.reshape(Batchsize, N_point, -1) # B, N, C


            mask_local_ = mask_local.reshape(Batchsize, L_sub_clip, -1, self.token_dim)
            mask_local_list = torch.split(tensor=mask_local_, split_size_or_sections=1, dim=1) # S-1*[B, 1, N', C]
            mask_local_list = [torch.squeeze(input=mask_local_frame, dim=1).contiguous() for mask_local_frame in mask_local_list] # S-1*[B, N', C]
            mask_local_recontrust = []
            nn = mask_local_list[0].size()[1]

    
            mask_global_list = []
            output_global = output_global.reshape(Batchsize, Sub_clips-1, N_out, self.token_dim) # B, S-1, N, C
            for i in range(Sub_clips-1):
                output_g = output_global[:,i,:,:].reshape(-1, self.token_dim)
                mask_global = self.mlp_head_global(output_g) # B, N, C
                mask_global = mask_global.reshape(Batchsize, -1, self.token_dim)
                mask_global = torch.max(input=mask_global, dim=1, keepdim=True)[0]  # B, 1, C
                mask_global_list.append(mask_global.repeat(1, nn, 1))
            
            for i in range(len(mask_local_list)):
                
                mask_local_global = torch.cat((mask_local_list[i], mask_global_list[i]), dim=2)     # [B, N, 2C]
                mask_local_global = mask_local_global.repeat(1, 64, 1)

                clip_anchor_i = clip_anchor[i]
                clip_anchor_i = clip_anchor_i.repeat(1,4,1)
                clip_anchor_i = self.fc(clip_anchor_i)
                
                pos_ = self.absolute_pos_embed[i].unsqueeze(0).unsqueeze(0)
                
                mask_local_global = mask_local_global + pos_
                mask_local_global_copy = mask_local_global
                mask_local_global = torch.cat((clip_anchor_i, mask_local_global),dim=2) # B, N, 2C+3
                
                mask_xyz = self.Fold1(mask_local_global) # B, N, 6
                mask_xyz = torch.cat((mask_xyz, mask_local_global_copy), dim=-1)
                mask_xyz = self.Fold2(mask_xyz)                           # [B, N, 6]
                mask_local_recontrust.append(mask_xyz)
            mask_xyz = torch.stack(mask_local_recontrust, dim=1).reshape(Batchsize, -1, C_xyz+3)

            # feature interpolate
            emb_xyzs_last = emb_xyzs[:,-1,:,:].contiguous()
            dist, idx = pointnet2_utils.three_nn(label_xyz.contiguous(), emb_xyzs_last) # (anchor, neighbor)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            mask_local = mask_local.transpose(1,2) # [B, C, N]
            mask_local = pointnet2_utils.three_interpolate(mask_local.contiguous(), idx, weight) # [B, C, N]
            mask_local = mask_local.transpose(1,2) # [B, N, C]

            label_global = F.normalize(label_global, dim=-1)  # [B, C]
            cls_token_global = F.normalize(cls_token_global, dim=-1)    # [B, C]

            mask_local = mask_local.reshape((-1, C_out))      # [B*N, C]
            mask_local = F.normalize(mask_local, dim=-1)
            label_local = label_local.reshape((-1, C_out))    # [B*N, C]
            label_local = F.normalize(label_local, dim=-1) # B, C
            label_local = torch.cat((label_local, negative_extra),dim=0) # B(S-1)N, C
            score_local = torch.matmul(mask_local, label_local.transpose(0,1)) #  B, B


            score_local = score_local / self.temperature
            target_sim_local = torch.arange(score_local.size()[0]).to(device)
            loss_local = self.criterion_local(score_local, target_sim_local)
         
            dist1, dist2 = self.criterion_dist(mask_xyz, clip2_xyz_rgb)
            loss_dist = torch.mean(dist1) + torch.mean(dist2)

            loss = loss_local + 30*loss_dist

            acc1_local, acc5_local = utils.accuracy(score_local, target_sim_local, topk=(1, 5))

            return loss, loss_local, loss_dist,cls_token_global, label_global, acc1_local, acc5_local 
                   # B,C          # B,C
        else:
            new_features = new_features.permute(0, 1, 3, 2)      # [B, L, N, C] 

            BS, L_out, N_out, C_out = new_features.shape
            new_features = new_features.reshape(Batchsize, NUM_CLIPS,L_out,N_out,C_out)
           
            output = new_features.reshape((Batchsize, NUM_CLIPS, N_out, C_out)) # [B, L, N, C]

            output = torch.mean(input=output, dim=-2, keepdim=False)    # (B, L, C)
            output = torch.max(input=output, dim=1, keepdim=False)[0]   # (B, C)
            
            output = self.fc_out0(output)

            return output
