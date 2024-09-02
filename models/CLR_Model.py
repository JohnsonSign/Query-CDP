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

        self.query = nn.Parameter(torch.randn(1,64, self.token_dim))

        self.pos_embedding = nn.Conv1d(
            in_channels=4, 
            out_channels=self.token_dim, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        self.pos_embedding_query = nn.Conv1d(
            in_channels=4, 
            out_channels=self.token_dim, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
       

        self.transformer = Transformer(
            self.token_dim, 
            self.depth, 
            self.heads, 
            self.dim_head, 
            self.mlp_dim
        )

        self.transformer_decoder = Transformer(
            self.token_dim, 
            1, 
            self.heads, 
            self.dim_head, 
            self.mlp_dim
        )
        self.mlp_decoder = nn.Linear(self.token_dim, 64, bias=False)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim, bias=False),
            nn.BatchNorm1d(self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, self.mlp_dim, bias=False),
            nn.BatchNorm1d(self.mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_dim, self.token_dim,),
        )

    
        self.absolute_pos_embed = nn.Parameter(torch.randn(4, self.token_dim))

        self.pretraining = pretraining
        if self.pretraining:   
            self.temperature = temperature

            self.mask_temporal = nn.Sequential(
                nn.Linear(self.token_dim, 4, bias=False),
                nn.BatchNorm1d(4),
                nn.ReLU(inplace=True),)
            
            self.new_recover_loss = nn.SmoothL1Loss(reduction='mean')
            self.criterion_global = torch.nn.CrossEntropyLoss()
            self.criterion_local = torch.nn.CrossEntropyLoss()
            self.t_loss = torch.nn.CrossEntropyLoss()
            

        else:

            self.fc_out0 = nn.Linear(2048, num_classes)


    def forward(self, xyzs):

        device = xyzs.get_device()

        if self.pretraining:
            Batchsize, Sub_clips, L_sub_clip, N_point, C_xyz = xyzs.shape  # [B, S, L, N, 3] B: for one gpu
            xyzs = xyzs.reshape((-1, L_sub_clip, N_point, C_xyz)) # [B*S-1, L, N, 3]

        # backbone
        if not self.pretraining:
            Batchsize = xyzs.size()[0]
            xyzs = torch.split(xyzs,4,dim=1)   # 4*[B,L',N,3]
            NUM_CLIPS = len(xyzs)
            xyzs = torch.stack(xyzs,dim=1)     # [B,4,L',N,3]
            xyzs = xyzs.reshape((-1,xyzs.size()[2],xyzs.size()[3],xyzs.size()[4]))

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)
        num_token = new_features.size()[-1]
        new_xys_tar = new_xys.reshape(Batchsize, Sub_clips, L_sub_clip, num_token, 3)[:, -1].detach()
        new_recover_target = new_features.permute(0, 1, 3, 2).reshape(Batchsize, Sub_clips, L_sub_clip, num_token, -1)[:,-1].detach()
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
            
            query_tokens = repeat(self.query, '() n d -> b n d', b = Batchsize) # B, 64, C
            emb_tokens = torch.cat((emb_tokens, query_tokens), dim=1) # B, S-1*N + 64, C

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

            xyzts_query_token = torch.ones((xyzts.size()[0], N_out, 1), dtype=torch.float32, device=device)*cls_token_T
            xyzts_query_token = torch.cat((label_xyz, xyzts_query_token), dim=2) # B, N, 4

            xyzts = torch.reshape(input=xyzts, shape=(xyzts.shape[0], xyzts.shape[1]*xyzts.shape[2], xyzts.shape[3]))  # [B, S-1*N, 4]

            xyzts = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1) # [B, S-1*N+1, C]
            xyzts_query_token = self.pos_embedding_query(xyzts_query_token.permute(0, 2, 1)).permute(0, 2, 1)
            xyzts = torch.cat((xyzts, xyzts_query_token), dim=1) # B, S-1*N+1, C

            embedding = xyzts + emb_tokens  # [B, S-1*N+1, C]

            if self.emb_relu:
                embedding = self.emb_relu(embedding)

            # transformer
            output = self.transformer(embedding) # [B, S-1*N+1, C]
            output = output.reshape(-1, output.size()[-1]) # B*((S-1)*N+1), C
            output = self.mlp_head(output)       # B*((S-1)*N+1), C

            output = output.reshape((Batchsize, -1, self.token_dim)) # B, S-1*N+1, C
            output, query_token = torch.split(output, [(Sub_clips-1)*N_out, 64], dim=1) # B, S-1*N, C ; B, 1, C
            output = output.reshape((Batchsize, Sub_clips-1, N_out, -1))
            mask_local = query_token

            len_tar = new_recover_target.size()[1]
            mask_local_list = [mask_local for i in range(len_tar)] #

            new_recover_all = []
            for i in range(len(mask_local_list)):
                
                mask_local_global = mask_local_list[i].repeat(1, 8, 1)
                pos_ = self.absolute_pos_embed[i].unsqueeze(0).unsqueeze(0)
                mask_local_global = mask_local_global + pos_

                
                mask_local_global = self.transformer_decoder(mask_local_global)
                mask_local_global = self.mlp_decoder(mask_local_global)
               
                loss_frame = self.new_recover_loss(mask_local_global, new_recover_target[:,i])
            
                new_recover_all.append(loss_frame)

           
            mask_temporal_all = []
            for i in range(len(mask_local_list)):
                
                mask_local_global = mask_local_list[i].repeat(1,8,1)
                pos_ = self.absolute_pos_embed[i].unsqueeze(0).unsqueeze(0)
                mask_local_global = mask_local_global + pos_
                mask_local_global = mask_local_global.reshape(-1, mask_local_global.size()[-1])
                mask_temporal = self.mask_temporal(mask_local_global)           # [B, N, 4]
                mask_temporal = mask_temporal.reshape(Batchsize, -1, len_tar)
                mask_temporal_all.append(mask_temporal)
            mask_temporal = torch.stack(mask_temporal_all, dim=1).reshape(Batchsize, -1, len_tar) # B, L*N, 4


            mask_local = mask_local.reshape((-1, C_out))      # [B*N, C]
            mask_local = F.normalize(mask_local, dim=-1)
            label_local = label_local.reshape((-1, C_out))    # [B*N, C]
            label_local = F.normalize(label_local, dim=-1) # B, C
            label_local = torch.cat((label_local, negative_extra),dim=0) # B(S-1)N, C
            score_local = torch.matmul(mask_local, label_local.transpose(0,1)) #  B, B


            score_local = score_local / self.temperature
            target_sim_local = torch.arange(score_local.size()[0]).to(device)
            loss_local = self.criterion_local(score_local, target_sim_local)
         
        
            clip_index_tar = [torch.ones((new_recover_target.size()[0], new_recover_target.size()[2], 1), dtype=torch.float32, device=device)*i for i in range(len_tar)]
            clip_index_tar = torch.stack(clip_index_tar, dim=1)
            t_loss = self.t_loss(mask_temporal.reshape(-1,len_tar), clip_index_tar.reshape(-1).long())

            loss = loss_local + sum(new_recover_all) + t_loss

          
            acc1_local, acc5_local = utils.accuracy(score_local, target_sim_local, topk=(1, 5))
            
            return loss, loss_local, acc1_local, acc5_local 

        else:
            
            new_features = new_features.permute(0, 1, 3, 2)      # [B, L, N, C] 
    
            BS, L_out, N_out, C_out = new_features.shape
        
            new_features = new_features.reshape(Batchsize, NUM_CLIPS,L_out,N_out,C_out)
       
            output = new_features.reshape((Batchsize, NUM_CLIPS, N_out, C_out)) # [B, L, N, C]

            output = torch.mean(input=output, dim=-2, keepdim=False)    # (B, L, C)
            output = torch.max(input=output, dim=1, keepdim=False)[0]   # (B, C)
            
            output = self.fc_out0(output)

            return output
