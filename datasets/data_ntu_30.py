import os

path_10 = '/ml_workspace/yckj3949/code/sxx2/0425-Point-sequences-unsupervised-dp-global-msr/datasets/ntu60_train_10_eval_10.list'
path_30 = '/ml_workspace/yckj3949/code/sxx2/0425-Point-sequences-unsupervised-dp-global-msr/datasets/ntu60_train_30_eval_30.list'
path_all = '/ml_workspace/yckj3949/code/sxx2/0425-Point-sequences-unsupervised-dp-global-msr/datasets/ntu60.list'

Cross_Subject = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]


cross_subject_train_num_class = {}
cross_subject_test_num_class = {}
num_train = 0
num_test = 0
with open(path_all, 'r') as f:

    for line in f:
        name, nframes = line.split()
        subject = int(name[9:12])
        label = int(name[-3:]) - 1

        if subject in Cross_Subject:
            if label not in cross_subject_train_num_class:
                cross_subject_train_num_class[label]=1
            else:
                cross_subject_train_num_class[label]+=1
        else:
            if label not in cross_subject_test_num_class:
                cross_subject_test_num_class[label]=1
            else:
                cross_subject_test_num_class[label]+=1

cross_subject_train_num_class_10 = {}
cross_subject_test_num_class_10 = {}

with open(path_10, 'r') as f:

    for line in f:
        name, nframes = line.split()
        subject = int(name[9:12])
        label = int(name[-3:]) - 1

        if subject in Cross_Subject:
            if label not in cross_subject_train_num_class_10:
                cross_subject_train_num_class_10[label]=1
            else:
                cross_subject_train_num_class_10[label]+=1
        else:
            if label not in cross_subject_test_num_class_10:
                cross_subject_test_num_class_10[label]=1
            else:
                cross_subject_test_num_class_10[label]+=1

data_10_not_include = set(cross_subject_train_num_class.keys()) - set(cross_subject_train_num_class_10.keys())
print(data_10_not_include)

print(cross_subject_train_num_class_10)






