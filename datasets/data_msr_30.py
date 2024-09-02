
root = '/home/data-vol-2/137data/home/yckj3949/data/MSRAction/processed_data'
root_30 = '/home/data-vol-2/137data/home/yckj3949/data/MSRAction/processed_data_25_train_all_cate'

import os
import shutil
import random
num_train = 0
num_test = 0
dst_train = 0
dst_test = 0
root_dirs = os.listdir(root)
random.shuffle(root_dirs)


train_class_num = {2: 12, 8: 15, 15: 15, 3: 11, 13: 15, 7: 15, 6: 14, 4: 12, 19: 15, 9: 15, 1: 12, 18: 15, 17: 15, 11: 15, 10: 15, 16: 15, 0: 12, 12: 15, 14: 5, 5: 12}
train_class_num_30 = {key:int(value*0.25)+1 for key,value in train_class_num.items()}
train_class_num_30_find = {}
for video_name in root_dirs:
    if  (int(video_name.split('_')[1].split('s')[1]) <= 5):

        label = int(video_name.split('_')[0][1:])-1
        if label not in train_class_num_30_find:
            train_class_num_30_find[label]=1
            dst_file = os.path.join(root_30,video_name)
            src_file = os.path.join(root, video_name)
            shutil.copy(src_file, dst_file)
        else:
            if train_class_num_30_find[label]<train_class_num_30[label]:
                train_class_num_30_find[label]+=1
                dst_file = os.path.join(root_30,video_name)
                src_file = os.path.join(root, video_name)
                shutil.copy(src_file, dst_file)

    else:
        dst_file = os.path.join(root_30,video_name)
        src_file = os.path.join(root, video_name)
        shutil.copy(src_file, dst_file)
        

for video_name in os.listdir(root_30):
    if (int(video_name.split('_')[1].split('s')[1]) <= 5):
        dst_train+=1
    else:
        dst_test+=1

print('dst_train: ',dst_train) # 270
print('dst_test: ',dst_test) # 297





