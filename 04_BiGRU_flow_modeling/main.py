import os
import random
import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import Data_Split as DS
from model import BiGRU
from Mydataset import Mydataset


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_channel', metavar='INPUT', type=int, default=16)  # 输入通道数量
    parser.add_argument('-in_ker_num', metavar='INPUT', type=int, default=64)  # 输入内核数量
    parser.add_argument('-layers', metavar='INPUT', type=int, default=4)  # 网络层数
    parser.add_argument('-seq_len',  metavar='INPUT',type=int, default=16)  # 序列长度
    parser.add_argument('-ker_size',  metavar='INPUT',type=int, default=13)  # 内核大小
    parser.add_argument('-fold', metavar='INPUT', type=int, default=0)  # 折叠数
    parser.add_argument('-CUDA', metavar='INPUT', type=str, default='3')  # CUDA 版本
    parser.add_argument('-epochs', metavar='INPUT', type=int, default=100)  # 迭代训练次数
    parser.add_argument('-batch_size', metavar='INPUT', type=int, default=256)  # 批处理大小，每次迭代中使用的样本数
    parser.add_argument('-model_name', metavar='INPUT', type=str, default='./GRU_in_channel_{}-in_kernel_{}-layers_{}-seq_len_{:0>2d}-ker_size_{:0>2d}-fold_{}')  # 输出模型名称
    return parser.parse_args()


args = get_args()  # 获取参数
max_acc=0  # 最大准确率
output_size = 28  # 输出大小

dropout = 0.35  # dropout值  
save_dir = './save_models/'  #  保存模型文件夹
if not os.path.exists(save_dir): os.mkdir(save_dir)  # 没有就新建一个
vocab_text_size = 1500  # 语料库中词汇表的大小


input_channel = args.input_channel  # 输入通道数量  
num_channels = [args.in_ker_num] * args.layers  #  每一层内核数量
layers = args.layers  # 层数
num_packs = args.seq_len  # 序列长度
seq_leng = num_packs  # 序列长度
kernel_size = args.ker_size  # 内核大小
fold = args.fold  # 折叠数
CUDA_VISIBLE_DEVICES = args.CUDA  # CUDA版本
num_epochs = args.epochs   # 迭代次数
outname = args.model_name.format(input_channel, args.in_ker_num, args.layers, args.seq_len, kernel_size, fold)  # 模型输出
batch_size = args.batch_size  # 批处理大小

file_dir = "./new_data/"  # 文件夹路径
data = DS.main(file_dir, num_packs)  # [APP代码, 序列]

def scramble_data(text):  # 打乱数据
    '''
    参数:
    text: [[int]]
    返回: 
    - text[0]: 
    '''
    cc = list(zip(text))  # 创建一个包含文本中原始字符的图元列表；zip为制作元组
    random.seed(100)  # 随机种子
    random.shuffle(cc)  # 随机每个列表
    text[:] = zip(*cc)  # 解压为列表
    return text[0]

da = scramble_data(data)  # 数据打乱
x=[]
for j in range(5):  # 将 da 分成五等份切片，分为训练集、验证集、测试集
    x.append(da[(len(da)*j)//5: (len(da)*(j+1))//5])
train=[]
valid=x[fold%5]  # 从 x 中取出一个部分并赋值给验证集
test=x[(fold+1)%5]  # 从 x 中取出另一个部分并赋值给测试集
for i in range(2,5):  # 取出剩余部分分给训练集
    train+=x[(i+fold)%5]

train_x=np.array(train,'int32')[:,1:]  # 自变量,16序列
train_y=np.array(train,'int32')[:,0]  # 因变量，APP编号
valid_x=np.array(valid,'int32')[:,1:]  
valid_y=np.array(valid,'int32')[:,0]  
test_x=np.array(test,'int32')[:,1:]  
test_y=np.array(test,'int32')[:,0]  
new_text=[]  # [训练集自变量, 训练集因变量, 验证集自变量, 验证集因变量, 测试集自变量, 测试集因变量]
new_text.append(list(train_x))
new_text.append(list(train_y))
new_text.append(list(valid_x))
new_text.append(list(valid_y))
new_text.append(list(test_x))
new_text.append(list(test_y))


def data_load(new_data,seq_leng):  # 将数据转为数据加载器，每次可以自动抛出随机几个样本
    '''
    参数:
    - new_data: [训练集自变量, 训练集因变量, 验证集自变量, 验证集因变量, 测试集自变量, 测试集因变量]
    - seq_leng: 序列长度
    返回:
    - train_dataloader: 训练集加载器
    - valid_dataloader: 验证集加载器
    - test_dataloader: 测试集加载器
    '''
    train_dataloader = []  # 训练数据加载器
    valid_dataloader = []  # 验证数据加载器
    test_dataloader = []  # 测试数据加载器
    train_dataloader.append(
        DataLoader(Mydataset(np.array(new_data[0], dtype="int32"), np.array(new_data[1], dtype="int32"),seq_leng),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    valid_dataloader.append(
        DataLoader(Mydataset(np.array(new_data[2], dtype="int32"), np.array(new_data[3], dtype="int32"),seq_leng),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    test_dataloader.append(
        DataLoader(Mydataset(np.array(new_data[4], dtype="int32"), np.array(new_data[5], dtype="int32"),seq_leng),
                    batch_size=batch_size,
                    shuffle=True, num_workers=0))
    return [train_dataloader,valid_dataloader,test_dataloader]


def train(model, device, train_loader, optimizer, epochs, i, loss_fn):  # 
    '''
    参数:
    - model: 模型
    - device: 设备
    - train_loader: 训练数据集
    - optimizer: 优化器
    - epochs: 迭代次数
    - i: 折叠数
    - loss_fn: 损失值
    返回:
    - null
    '''
    model.train()  # 启用训练模式
    total_loss = 0  # 初始化总损失
    y_true = torch.LongTensor(0).to(device)  # 初始化预测标签的张量
    y_predict = torch.LongTensor(0).to(device)  # 初始化预测标签的张量
    correct = 0.   # 初始化正确样本数
    sum_num = 0.   # 初始化总样本数
    for idx, (data, target) in enumerate(train_loader):  # 遍历训练数据
        data, target = data.to(device), target.to(device)  # 将数据和标签复制到指定设备
        pre = model(data)  # 使用模型预测
        y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)  # 将预测结果与之前的结果拼接
        y_true = torch.cat([y_true, target], 0)  # 将真实标签与之前的结果拼接
        loss = loss_fn(pre.to(device), target.to(device)).to(device)  # 计算损失
        optimizer.zero_grad()  # 清空模型梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        
        total_loss += loss.item() * len(target)  # 累加损失
        
        sum_num += len(target)  # 累加样本数
        if idx % 100 == 99:  # 每进行100次训练，打印训练信息
            print("Fold {} Train Epoch: {}, iteration: {}, Loss: {}".format(i, epochs, idx + 1, loss.item()))
    avg_loss = total_loss / sum_num  # 计算平均损失
    
    y_true = y_true.cpu().numpy().tolist()  # 将真实标签转为numpy数组
    y_predict = y_predict.cpu().numpy().tolist()  # 将预测标签转为numpy数组

    y_true_trans = np.array(y_true)  # 数组转置
    y_predict_trans = np.array(y_predict)

    acc = balanced_accuracy_score(y_true_trans, y_predict_trans)  # 计算平衡精度
    train_acc = 100. * acc  # 计算训练精度

    print("---------------------------------------------------")  # 打印训练信息
    print("Fold {} epoch:{}  training Loss:{:.4f} training acc:{:.4f}".format(i, epoch, avg_loss, train_acc))


def valid(model, device, dev_loader, epoch, i, loss_fn , num_epochs, scheduler):  # 评估机器学习模型在验证集上性能
    '''
    参数:
    - model: 
    - device: 
    - dev_loader: 
    - epoch: 
    - i: 
    - loss_fn: 
    - num_epochs: 
    - scheduler: 
    返回:
    - acc: 
    '''
    model.eval()  # 将模型设置为评估模式
    y_true = torch.LongTensor(0).to(device)  # 创建空的真实标签张量并移动到指定设备上
    y_predict = torch.LongTensor(0).to(device)  # 创建空的预测标签张量并移动到指定设备上
    total_loss = 0.  # 初始化总损失
    correct = 0.  # 初始化正确率
    sum_num = 0.  # 验证集中的样本总数
    with torch.no_grad():  # 在评估模式下遍历验证集数据
        for idx, (data, target) in enumerate(dev_loader):
            data, target = data.to(device), target.to(device)  # 将数据和标签移动到指定设备上
            target = target.squeeze()  # 将目标的维度降为一维
            
            
            pre = model(data)  # 对数据运行模型，得到预测
            y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)  # 更新预测标签张量
            y_true = torch.cat([y_true, target], 0)  # 更新真实标签张量
            loss = loss_fn(pre, target)  # 计算损失
            total_loss += loss.item() * len(target)  # 计算总损失
            sum_num += len(target)  # 累加总损失样本数量
        avg_loss = total_loss / sum_num  # 计算平均损失
        y_true = y_true.cpu().numpy().tolist()  # 将真实标签张量移动到CPU上并转化为列表
        y_predict = y_predict.cpu().numpy().tolist()  # 将预测标签张量移动到CPU上并转化为列表

        y_true_trans = np.array(y_true)  # 将真实标签列表转化为numpy数组
        y_predict_trans = np.array(y_predict)  # 将预测标签列表转化为numpy数组

        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)  # 计算平衡准确率
        valid_acc = 100. * acc  # 计算验证集平衡准确率
    scheduler.step(avg_loss)  # 根据平均损失更新学习率
    print("Fold {} epoch:{} valid Loss:{:.4f} valid acc:{:.4f}".format(i, epoch, avg_loss, valid_acc))  # 输出验证集平均损失和平衡准确率
    print("---------------------------------------------------")
    return acc  # 返回平衡准确率


def test(device, test_loader, i,loss_fn,save_dir,outname):
    '''
    参数:
    - device: 
    - test_loader: 
    - i: 
    - loss_fn: 
    - save_dir: 
    - outname: 
    返回:
    - null
    '''
    
    model = TCN(input_channel=input_channel, output_size=output_size, num_channels=num_channels, kernel_size=kernel_size,
                dropout=dropout,
                vocab_text_size=vocab_text_size,seq_leng=seq_leng).to(device)  # 加载TCN模型
    
    model_dir = save_dir + "{}.pth".format(outname)  # 计算模型文件位置
    model.load_state_dict(torch.load(model_dir))  # 加载模型状态函数
    model.eval()  # 将模型设置为评估模式
    y_true = torch.LongTensor(0).to(device)  # 初始化真实值的tensor
    y_predict = torch.LongTensor(0).to(device)  # 初始化预测值的tensor
    total_loss = 0.  # 初始化总损失
    correct = 0.  # 初始化正确样本数
    sum_num = 0.
    with torch.no_grad():  # 遍历整个测试集
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)  # 将数据和目标转移到设备上
            pre = model(data)  # 应用模型并获取预测
            y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)  # 将预测值添加到 y_predict tensor 中
            y_true = torch.cat([y_true, target], 0)  # 将真实值添加到 y_true tensor 中
            target = target.squeeze()  # 从目标中提取单个维度
            loss_fn = loss_fn.to(device)  # 将损失函数转移到设备上
            loss = loss_fn(pre, target)  # 使用损失函数计算损失
            total_loss += loss.item() * len(target)  # 累加损失
            sum_num += len(target)  # 累加样本数
        avg_loss = total_loss / sum_num  # 计算平均损失
        y_true_list = y_true.cpu().numpy().tolist()  # 将真实值从 tensor 中提取出来转换为列表
        y_predict_list = y_predict.cpu().numpy().tolist()  # 将预测值从 tensor 中提取出来转换为列表

        y_true_trans = np.array(y_true_list)  # 将列表转换为 numpy 数组
        y_predict_trans = np.array(y_predict_list)
        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)  # 使用平衡准确率度量准确率
        test_acc = 100. * acc  # 计算百分比准确率
    print("Fold {} test Loss:{:.4f} test acc:{:.4f}".format(i, avg_loss,test_acc))  # 打印出损失和准确率
    print("---------------------------------------------------")


if __name__ == '__main__':
    
    dataload = np.squeeze(data_load(new_text,seq_leng))  # [训练集数据加载器, 验证集数据加载器, 测试集数据加载器]
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES  # 设置CUDA
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")  # 调用GPU
    
    model = BiGRU(input_channel, kernel_size, layers, output_size).to(device)  # 创造TCN模型
    
    criterion = torch.nn.CrossEntropyLoss()  # 损失函数
    
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=5e-3, weight_decay=0)  # 使用默认参数的Adam优化器，权重衰减为0之外
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, verbose=True, min_lr=1e-5)  # 如果验证准确率没有提高，使用调度器来降低学习率
    max_acc = 0  # 掌握最大验证精度
    for epoch in range(10):#range(num_epochs):  # 对模型进行指定数量的迭代训练
        train(model, device, dataload[0], optimizer, epoch, int(fold)+1,criterion)  # 训练
        valid_acc = valid(model, device, dataload[1], epoch, int(fold)+1,criterion, num_epochs, scheduler)  # 在验证集上评估模型，并根据验证精度保存最佳模型
        if max_acc < valid_acc:  #
            max_acc = valid_acc
            torch.save(model.state_dict(), save_dir + "{}.pth".format(outname))
    # test(device, dataload[2], int(fold)+1, criterion, save_dir, outname)  # 在测试集上评估最佳模型
    