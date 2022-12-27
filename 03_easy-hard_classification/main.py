import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from model1 import TCN
from Mydataset import Mydataset
import Get_filename_1 as GF


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_channel', metavar='INPUT', type=int, default=64)  # 输入通道数量
    parser.add_argument('-in_ker_num', metavar='INPUT', type=int, default=64)  # 输入内核数量
    parser.add_argument('-layers', metavar='INPUT', type=int, default=4)  # 网络层数
    parser.add_argument('-seq_len',  metavar='INPUT',type=int, default=32)  # 序列长度
    parser.add_argument('-ker_size',  metavar='INPUT',type=int, default=13)  # 内核大小
    parser.add_argument('-fold', metavar='INPUT', type=int, default=0)  # 折叠数
    parser.add_argument('-CUDA', metavar='INPUT', type=str, default='3')  # CUDA 版本
    parser.add_argument('-batch_size', metavar='INPUT', type=int, default=256)  # 迭代训练次数
    parser.add_argument('-model_name', metavar='INPUT', type=str, default='./save_models/in_channel_64-in_kernel_064-layers_4-seq_len_32-ker_size_13-fold_0.pth')  # 批处理大小，每次迭代中使用的样本数
    parser.add_argument('-input_name', metavar='INPUT', type=str, default='decision-entropy-ntree_30-ratio_0.8-pack_4-fold_3')  # 输出模型名称
    return parser.parse_args()


args = get_args()  # 获取参数
output_size = 28  # 输出大小
max_words = 10000  
dropout = 0.35  # dropout值    
save_dir = './save_models/'   #  保存模型文件夹 
vocab_text_size = 1500   # 语料库中词汇表的大小
CUDA_VISIBLE_DEVICES=args.CUDA  # CUDA版本
input_channel = int(args.input_channel)  # 输入通道数量
num_channels = [int(args.in_ker_num)] * int(args.layers)  # 每一层内核数量  
num_packs =int(args.seq_len)  # 序列长度
seq_leng =num_packs   # 序列长度
kernel_size = int(args.ker_size)  # 内核大小  
fold = args.fold  # 折叠数
outname = args.model_name   # 输出名称
input_name=args.input_name  # 输入名称


def read_data(file_dir,seq_leng):  # 读取第一阶段生成的文件[验证集自变量, 验证集标签, 测试集自变量, 测试集标签]
    '''
    参数:
    - file_dir: 存储文件目录
    - seq_leng: 序列长度
    返回:
    - valid_data: 验证集自变量
    - valid_labels: 验证集标签
    - test_data: 测试集自变量
    - test_labels: 测试集标签
    '''
    global valid_pre_labels,test_pre_labels,valid_true_labels,test_true_labels
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []

    birth_data1=[]
    with open(file_dir + input_name +"_valid_pre_true.csv") as csvfile1:  # 验证集[预测标签，真实标签]
        tempvalid = csv.reader(csvfile1)
        for row in tempvalid:  
            birth_data1.append(row)
    valid_pre_labels=np.array(birth_data1,"int32")[:,0]  # 验证集预测标签
    valid_true_labels = np.array(birth_data1,"int32")[:, 1]  # 验证集真实标签

    birth_data2=[]
    with open(file_dir + input_name +"_test_pre_true.csv") as csvfile2:  # 测试集[预测标签, 真实标签]
        temptest = csv.reader(csvfile2)
        for row in temptest:
            birth_data2.append(row)
            
    test_pre_labels=np.array(birth_data2,"int32")[:,0]  # 测试集预测标签
    test_true_labels = np.array(birth_data2,"int32")[:, 1]  # 测试集真实标签


    birth_data3=[]
    with open(file_dir + input_name +"_valid_test_data.csv") as csvfile3:  # 未完成分类的[[验证集自变量, 验证集因变量, 测试集自变量, 测试集因变量]]
        tempdata = csv.reader(csvfile3)
        for row in tempdata:
            birth_data3.append(row)
    temp_valid_data=np.array(birth_data3)[0]  # 验证集自变量
    for id1, e in enumerate(temp_valid_data):
        try:
            e1 = e.strip('[]').replace("'", "").replace(" ", "").split(',')
        except:
            break
        temp1 = []
        for i in range(seq_leng):
            if int(e1[3 * i])<0:
                temp1.append(0)
            else:
                temp1.append(int(e1[3 * i]))
        valid_data.append(temp1)  # 验证集自变量
        valid_labels.append(int(birth_data3[1][id1]))  # 验证集标签
    temp_test_data=np.array(birth_data3)[2]  # 测试集自变量
    for id2, e in enumerate(temp_test_data):  # 测试集标签
        try:
            e1=e.strip('[]').replace("'","").replace(" ","").split(',')
        except:
            break
        temp1 = []
        for i in range(seq_leng):
            if int(e1[3 * i])<0:
                temp1.append(0)
            else:
                temp1.append(int(e1[3 * i]))
        test_data.append(temp1)
        test_labels.append(int(birth_data3[3][id2]))
    return valid_data, valid_labels, test_data, test_labels


def data_load(valid_data,valid_labels,test_data,test_labels,seq_leng):  # 生成加载器
    '''
    参数:
    - valid_data: 验证集自变量
    - valid_labels: 验证集标签
    - test_data: 测试集自变量
    - test_labels: 测试集标签
    - seq_leng: 序列长度
    返回:
    - valid_dataloader: 验证集加载器
    - test_dataloader: 测试集加载器
    '''
    valid_dataloader = []
    test_dataloader = []
    valid_dataloader.append(
        DataLoader(Mydataset(np.array(valid_data, dtype="int32"), np.array(valid_labels, dtype="int32"),seq_leng),
                    batch_size=64,
                    shuffle=True, num_workers=0))
    test_dataloader.append(
        DataLoader(Mydataset(np.array(test_data, dtype="int32"), np.array(test_labels, dtype="int32"),seq_leng),
                    batch_size=64,
                    shuffle=True, num_workers=0))
    return [valid_dataloader,test_dataloader]


def valid(model, device, valid_loader, i,loss_fn):  # 
    '''
    参数:
    - model: 模型
    - device: 设备
    - valid_loader: 验证集加载器 
    - i: 折叠数
    - loss_fn: 损失函数
    返回: null
    '''
    global valid_pre_labels,valid_true_labels
    model.eval()  # 评估模式
    y_true = torch.LongTensor(0).to(device)  # 真实数据
    y_predict = torch.LongTensor(0).to(device)  # 预测数据
    total_loss = 0.  # 初始化总损失
    no_aprt=0  # 计算错误未通过的样本数
    correct = 0.  # 初始化正确率
    sum_num = 0.  # 验证集中的样本总数
    with torch.no_grad():  # 在评估模式下遍历验证集数据
        for idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)  # 将数据和标签移动到指定设备上
            try:
                pre = model(data)  # 运行模型，得到预测数据
                y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)  # 更新预测张量
                y_true = torch.cat([y_true, target], 0)  # 更新真实张量
                target = target.squeeze()
                loss_fn = loss_fn.to(device)
                loss = loss_fn(pre, target)  # 计算损失
                total_loss += loss.item() * len(target)  # 总损失
                sum_num += len(target)  # 验证集样本
            except:
                no_aprt = no_aprt + target.shape[0]
                print("[alter] {} samples didn't go through testing.".format(no_aprt))
        avg_loss = total_loss / sum_num  # 平均损失
        y_true_list = y_true.cpu().numpy().tolist()  # 真实标签tensor
        y_predict_list = y_predict.cpu().numpy().tolist()  # 预测标签tensor

        y_true_trans = np.array(y_true_list)  # 真实标签numpy
        y_predict_trans = np.array(y_predict_list)  # 预测标签numpy
        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)  # 准确率
        valid_acc = 100. * acc
        y_true_trans1 = np.array(y_true_list+list(valid_true_labels))
        y_predict_trans1 = np.array(y_predict_list+list(valid_pre_labels))
        acc1 = balanced_accuracy_score(y_true_trans1, y_predict_trans1)
        valid_acc1 = 100. * acc1
    print("Testing fold {} loss:{:.4f}, acc:{:.4f}".format(i, avg_loss, valid_acc))
    print("Testing fold {} current total loss:{:.4f} current total acc:{:.4f}".format(i, avg_loss,valid_acc1))
    print("---------------------------------------------------")


def test(model, device, test_loader, i, loss_fn):  # 
    '''
    参数:
    - model: 
    - device: 
    - valid_loader: 
    - i: 
    - loss_fn: 
    返回: null
    '''
    global test_pre_labels, test_true_labels
    model.eval()
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    total_loss = 0.
    no_aprt = 0
    correct = 0.
    sum_num = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            try:
                pre = model(data)
                y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
                y_true = torch.cat([y_true, target], 0)
                target = target.squeeze()
                loss_fn = loss_fn.to(device)
                loss = loss_fn(pre, target)
                total_loss += loss.item() * len(target)
                sum_num += len(target)
            except:
                no_aprt=no_aprt+target.shape[0]
                print("[alter] {} samples didn't go through testing.".format(no_aprt))

        avg_loss = total_loss / sum_num
        y_true_list = y_true.cpu().numpy().tolist()
        y_predict_list = y_predict.cpu().numpy().tolist()

        y_true_trans = np.array(y_true_list)
        y_predict_trans = np.array(y_predict_list)
        matrix = confusion_matrix(y_true_trans, y_predict_trans)
        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
        test_acc = 100. * acc

        y_true_trans1 = np.array(y_true_list+list(test_true_labels))
        y_predict_trans1 = np.array(y_predict_list+list(test_pre_labels))
        matrix1 = confusion_matrix(np.array(y_true_trans1), np.array(y_predict_trans1))
        acc1 = balanced_accuracy_score(y_true_trans1, y_predict_trans1)
        test_acc1 = 100. * acc1
    print("Valid fold {} loss:{:.4f} acc:{:.4f}".format(i, avg_loss,test_acc))
    print("Valid fold {} current total loss:{:.4f} current total acc:{:.4f}".format(i, avg_loss, test_acc1))
    print("---------------------------------------------------")



if __name__ == '__main__':
    file_dir = "data/"  # 文件所在目录
    valid_data, valid_labels, test_data, test_labels = read_data(file_dir, seq_leng)  # [验证集自变量, 验证集标签, 测试集自变量, 测试集标签]
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES  # cuda
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")  # 设备

    dataload = data_load(valid_data,valid_labels,test_data,test_labels,seq_leng)  # [验证集加载器, 测试集加载器]
    dataload = np.squeeze(dataload)  # [验证集加载器, 测试集加载器]

    model = TCN(input_channel=input_channel, output_size=output_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout,
                vocab_text_size=vocab_text_size, seq_leng=seq_leng).to(device)  # TCN模型

    model.load_state_dict(torch.load(outname, map_location=torch.device('cpu')))  # 加载模型参数

    criterion = torch.nn.CrossEntropyLoss()  # 损失函数

    valid(model, device, dataload[0], fold, criterion)
    test(model, device, dataload[1], fold, criterion)