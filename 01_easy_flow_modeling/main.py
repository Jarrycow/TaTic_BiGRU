import csv
import random
import argparse
from collections import Counter

import numpy as np
from sklearn import tree, model_selection
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt


import Data_Split as DS


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)  #  创建一个 ArgumentParser 对象，用于处理命令行参数
    parser.add_argument('-num_trees', metavar='INPUT', type=str, default='30')   # 决策树的数量
    parser.add_argument('-condition', metavar='INPUT', type=str, default='0.8')   # 分割节点的条件
    parser.add_argument('-packs', metavar='INPUT', type=str, default='4')  # 多线程每个线程的决策树数目
    parser.add_argument('-fold',metavar='INPUT', type=str, default='3')  # 交叉折叠数(将数据分为几份，其中1份作为测试集，其余作为训练集)
    parser.add_argument('-out_name', metavar='INPUT', type=str, default='outputs')  # 输出文件名
    parser.add_argument('-criterion', type=str, default='entropy',choices={'entropy', 'gini'})  # 决策树建立标准
    return parser.parse_args()

# 全局变量
args = get_args()  # 获得参数
file_dir = "./new_data/"  # 数据文件夹
fold_num = 5  # 交叉验证中折叠数

seq_leng = 96  # 序列长度
criterion = args.criterion  # 决策树构建标准  

num_trees = int(args.num_trees)  # 决策树树数量
fold= int(args.fold)  # 交叉折叠数
num_packs = int(args.packs)  # 模型包数
condition = int(num_trees*float(args.condition))  

out_name = '{}-{}-ntree_{:0>2d}-ratio_{}-pack_{}-fold_{}'.format(
    'decision', criterion, num_trees, args.condition, num_packs, fold)  # 输出文件-构建标准-ntree_决策树数量-ratio_策略-pack_线程数树数目-fold_交叉验证数目"



def scramble_data(text):
    cc = list(zip(text))
    random.seed(100)
    random.shuffle(cc)
    text = list(zip(*cc))
    return text[0]


def x_y_data(data):
    x=[]
    y=[]
    for i in data:
        temp = []
        for idx, j in enumerate(i[1], start=1):
            if idx % 3==0:
                j*=10000 
                if j < 1000 and j >= 0:
                    j = (j // 100 + 1)
                elif j < 10000 and j >= 1000:
                    j = (j // 50 + 1)
                elif j >= 10000:
                    j= (j // 10 + 1)
            temp.append(int(j))
        x.append(temp)
        y.append([i[2], i[0]])
    return x, y


def x_y(train, valid, test):
    x1,y1 = x_y_data(list(train))
    x2,y2 = x_y_data(list(valid))
    x3,y3 = x_y_data(list(test))
    return x1, y1, x2, y2, x3, y3


def format_division(text,fold_num):  # 分训练集、验证集、测试集
    '''
    参数:
    - text: 数据
    - fold_num: 数据包数量
    返回值:
    - train_x: 训练集自变量
    - train_y: 训练集因变量
    - valid_x: 验证集自变量
    - valid_y: 验证集因变量
    - test_x : 测试集自变量 
    - test_y : 测试集因变量
    '''
    all_data = scramble_data(text)
    x = []
    fold_size = (len(all_data) // fold_num)
    for j in range(fold_num):
        x.append(all_data[int(fold_size * j):int(fold_size * (j + 1))])
    train = []
    valid = x[int(fold) % fold_num]
    test = x[(int(fold) + 1) % fold_num]
    for i in range(2, 5):
        train += x[(i + int(fold)) % fold_num]

    train_x, train_y, valid_x, valid_y, test_x, test_y=x_y(train, valid, test)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

result_valid=[]  # 验证集结果
result_test=[]  # 测试集结果

def valid_test(clf,Xtest,Ytest,dot_data,data_id,criterion,flag):  # 返回本次决策树预测结果和真实结果
    '''
    参数:
    - clf: 决策树分类器
    - Xtest: 自变量
    - Ytest: 因变量
    - dot_data: 决策树
    - data_id: 代表的APP
    - criterion: 
    - flag:
    返回:
    - data_id: 代表的APP
    - pred_label: 预测结果
    - Ytest: 真实因变量
    '''
    global  result_valid, result_test    
    pred = clf.predict(Xtest)    # 用分类器去预测结果
    idex = clf.apply(Xtest)  # 第i个样本到达叶节点的索引
    if flag == "valid":
        result_valid.append(clf.score(Xtest, Ytest))  # score 评估模型预测的质量
    if flag == "test":
        result_test.append(clf.score(Xtest, Ytest))
    
    criterion_min = []
    for temp in dot_data.split(';'):  # 将DOT数据拆为单独行
        if '{} = 0.0'.format(criterion) in temp:  # 搜索包含criterion字符串和值 0.0 的行
            if '<=' not in temp:   # 若找到且不包含`<=`，该符号在搜索树中一般表示条件拆分
                new_temp = temp.replace('[', '').replace(']', '').split('\\n')  # 删除[]、\n
                criterion_value = new_temp[0].split(' = ')[-1]  # 
                sample_value = new_temp[1].split(' = ')[-1]  # 样本数据值
                if float(criterion_value) <= 0.00 and (int(sample_value))>0:
                    criterion_min.append([new_temp[0].split(' ')[0], criterion_value])
    dis = np.array(np.array(criterion_min)[:, 0], 'int32')
    pred_label = list(map(lambda x,y: y if x in dis else -1, idex, pred))
    return data_id, pred_label, Ytest

def max_label(x,condition):
    x=list(x)
    x_c = Counter(x)
    label, value = x_c.most_common(1)[0]
    if value >= condition:
        return label
    return -1

def build_tree_up_to_down(train_x, train_y, valid_x, valid_y, test_x, test_y,file_i,valid_id,test_id, condition, criterion='gini',num_trees=5):  # 树建立
    '''
    参数:
    - train_x: 训练集自变量, 12个数字的流
    - train_y: 训练集因变量, 代表 APP 的代码
    - valid_x: 验证集自变量
    - valid_y: 验证集因变量
    - test_x : 测试集自变量 
    - test_y : 测试集因变量
    - file_i : 交叉折叠数
    - valid_id: 验证集代表 APP
    - test_id: 测试集代表 APP
    - condition: 分支条件
    - criterion='gini'决策方法
    - num_trees: 决策树个数
    返回:
    - [valid_need_agains, test_need_agains]: 验证集、测试集下需要进一步分类的APP
    - valid_Tag_thrown_out: 验证集[(预测标签，真实标签)]
    - test_Tag_thrown_out: 训练集[(预测标签，真实标签)]
    - easy_flow_test: 
    '''
    global result_test,result_valid  # 测试集结果；验证集结果
    x=[]  # 存储训练集样本编号
    y=[]
    valid_need_agains=[]  # 未完成分类样本APP
    test_need_agains=[]  # 存储再次测试集样本编号

    for i in range(num_trees):  # 训练集交叉划分,可能是为了拥有更多的训练集
        print("num_trees: " + str(i) + '\t' + str(num_trees))
        x_train, _, y_train,_ = model_selection.train_test_split(train_x, train_y, random_state=random.randint(0,100000), test_size=0.368)  # 随机划分训练集和测试集
        x.append(x_train)  # 自变量、因变量加入训练集
        y.append(y_train)



    valid_need_again = []  # 验证集APP
    valid_pre_labels=[]  # 验证集预测标签列表
    test_need_again = []  # 测试集APP
    test_pre_labels = []  # 测试集预测标签编号
    for i in range(num_trees):  # 训练决策树
        print("num_trees: " + str(i) + '\t' + str(num_trees))
        clf = tree.DecisionTreeClassifier(criterion=criterion)  # 构建决策树分类器，不纯度的计算方法使用基尼系数
        clf = clf.fit(x[i], y[i])  # 拟合训练
        # dot_data = tree.export_graphviz(clf, out_file=str(i)+"tree.dot")  # 以 DOT 格式导出决策树
        dot_data = tree.export_graphviz(clf, out_file=None) # DOT格式保存生成的树
        
        valid_a,valid_b,valid_c=valid_test(clf,valid_x,valid_y,dot_data,valid_id,criterion,"valid")  # APP, 预测结果, 真实因变量
        valid_need_again=valid_a  # 存储真实的APP
        valid_pre_labels.append(valid_b)  # 验证集预测结果
        valid_real_labels=valid_c  # 验证集真实因变量

        test_a,test_b,test_c=valid_test(clf, test_x, test_y, dot_data, test_id, criterion,"test")  # 在测试集上进行预测并获取结果
        test_need_again = test_a  # 存储真实的APP
        test_pre_labels.append(test_b)    # 测试集预测结果
        test_real_labels = test_c  # 测试集真实因变量


    valid_temp=[]  # 
    pre2=[]  # 完成分类的预测标签
    tru2=[]  # 完成分类的真实标签
    valid_Tag_thrown_out=[]  # [(预测标签，真实标签)]
    temp_valid_pre_labels = np.array(valid_pre_labels).T  # 验证集预测结果转置
    valid_pre = list(map(lambda t: max_label(t, condition), temp_valid_pre_labels))  # 每个列表出现次数最多且多于condition标签，类似于综合每个搜索树的结果
    valid_values = np.array(valid_pre)  # 搜索森林预测结果向量
    idx_easy = valid_values > -1  # 完成分类的样本
    idx_hard = valid_values == -1  # 未完成分类的样本
    pre2 = np.array(valid_values)[idx_easy]  # 完成分类的预测标签
    tru2 = np.array(valid_real_labels)[idx_easy]  # 完成分类的真实标签
    valid_Tag_thrown_out=list(zip(pre2, tru2.astype(np.int32)))  # [(预测标签，真实标签)]
    valid_temp.extend(np.array(valid_need_again)[idx_hard])  # 未完成分类的样本标签
    valid_need_agains.append(valid_temp)  # 添加到“验证集需要再次分类的样本”
    pred2 = np.array(valid_real_labels)  
    pred2[idx_hard] = -1
    cov2 = balanced_accuracy_score(valid_real_labels, pred2)
    print("Fold {} valid Cov {:.4f}".format(file_i,cov2*100))
    acc2 = balanced_accuracy_score(tru2, pre2)  # 有标签样本时的平衡准确率
    print("Fold {} valid AoC {:.4f}".format(file_i,acc2*100))  # 分类器的平衡覆盖率

    test_temp = []  
    pre1=[]  
    tru1=[]  
    test_Tag_thrown_out = []  
    easy_flow_test = list()
    temp_test_pre_labels = np.array(test_pre_labels).T  
    test_pre = list(map(lambda t: max_label(t, condition), temp_test_pre_labels))  
    test_values = np.array(test_pre)  
    idx_easy = test_values > -1  
    idx_hard = test_values == -1  
    pre1 = test_values[idx_easy]  
    tru1 = test_real_labels[idx_easy]  
    test_Tag_thrown_out=list(zip(pre1, tru1.astype(np.int32)))  
    easy_flow_test.extend(np.array(test_need_again)[idx_easy])
    test_temp.extend(np.array(test_need_again)[idx_hard])  
    test_need_agains.append(test_temp)
    pred1 = np.array(test_real_labels)
    pred1[idx_hard] = -1
    cov1 = balanced_accuracy_score(test_real_labels, pred1)
    print("Fold {} test Cov {:.4f}".format(file_i,cov1*100))
    acc1 = balanced_accuracy_score(tru1, pre1)
    print("Fold {} test AoC {:.4f}".format(file_i,acc1*100))
    return [valid_need_agains, test_need_agains], valid_Tag_thrown_out, test_Tag_thrown_out, easy_flow_test


def main_fun(file_dir,num_packs,fold_num,criterion,num_trees,condition,fold):  # 
    '''
    参数:
    - file_dir: 数据文件夹
    - num_packs: 每个线程决策树数目
    - fold_num: 交叉验证折叠数
    - criterion: 决策树标准
    - num_trees: 决策树数量
    - condition: 分叉条件
    - fold:  ？
    返回:
    - need_again_id: 
    - long_text: 
    '''
    short_text, long_text = DS.main(file_dir,num_packs)  # 数据预处理（读取数据，按照APP分类，并且分类长短序列）；数据类型[APP, [序列(短25长96)], APP代码]
    print("data GET")
    train_x, train_y, valid_x, valid_y, test_x, test_y = format_division(short_text,fold_num)  # 分三个数据集
    print("train! valid! test!")
    need_again_id, valid_Tag, test_Tag, easy_flow_test =build_tree_up_to_down(train_x, np.array(np.array(np.array(train_y))[:, 0], 'int32'), valid_x,
                          np.array(np.array(np.array(valid_y))[:, 0], 'int32'), test_x,
                          np.array(np.array(np.array(test_y))[:, 0], 'int32'), fold,
                          np.array(np.array(valid_y))[:, 1], np.array(np.array(test_y))[:, 1], condition,
                          criterion, num_trees)  # 构建决策树

    with open('./needdata/{}_valid_pre_true.csv'.format(out_name), 'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(valid_Tag)  # 写入验证集[预测标签，真实标签]
    with open('./needdata/{}_test_pre_true.csv'.format(out_name), 'w', newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(test_Tag)  # 写入测试集[预测标签，测试标签]
    return need_again_id,long_text  # 返回需要进行下一步检测的


def read_all_data(temp): # 将列表第2列全部转为int
    '''
    参数:
    - temp: 列表
    返回:
    - new_text: 列表
    '''
    new_text=[]
    temp_data=np.array(temp)[:,0:2]
    for idx, e in enumerate(temp_data):
        e1=np.array(e[1],"int32")
        new_text.append([temp[idx][0],e1,temp[idx][-1]])
    return new_text

def length_same(text,leng):  # 将text中序列转为同样长度
    '''
    参数:
    - text: 较长的数据[APP, [序列], APP编号]
    - leng: 长度
    返回:
    - same_text: 将[序列]统一长度为leng
    '''
    same_text = []
    for length_list1 in text:
        if len(length_list1[1]) > leng:
            length_list1[1] = list(length_list1[1])[:leng]
        else:
            length_list1[1] = list(length_list1[1]) + [0,0,0] * ((leng - len(length_list1[1]))//3)
        same_text.append(length_list1[:3])
    return same_text

def need_text_label(same_text,idx):  # 
    '''
    参数:
    - same_text: 较长的数据[APP, [序列], APP编号]
    - idx: 未分类完毕的数据
    返回:
    - data: 
    '''
    data=[]
    valid_data = []
    valid_label = []
    test_data = []
    test_label = []
    for temp in same_text:
        if temp[0] in idx[0][0]:
            valid_data.append(temp[1])
            valid_label.append(temp[2])
        if temp[0] in idx[1][0]:
            test_data.append(temp[1])
            test_label.append(temp[2])
    data.append([valid_data,valid_label,test_data,test_label])
    return data

def main_fun1(long_text,id_valid_test,seq_leng):  # 
    '''
    参数；
    - long_text: 较长的数据[APP, [序列], APP编号]
    - id_valid_test: 第一阶段未分类完毕的数据
    - seq_leng: 包长度
    返回:
    - need_data: 
    '''
    long_text=read_all_data(long_text)  # 序列转为int
    long_text= length_same(long_text,seq_leng)  # 序列转为同样长度
    need_data = need_text_label(long_text, id_valid_test)
    return need_data

def save_need(need_data):  # 保存数据
    for i in range(len(need_data)):
        with open('./needdata/{}_valid_test_data.csv'.format(out_name), 'w',newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerows(need_data[i])

if __name__ == '__main__':    
    id_valid_test,long_text=main_fun(file_dir=file_dir,num_packs = num_packs,fold_num = fold_num,criterion =criterion,num_trees=num_trees,condition=condition,fold=fold)  # 未完成分类的标签
    
    need_data = main_fun1(long_text=long_text,id_valid_test=id_valid_test,seq_leng=seq_leng)  # 将未分类完成的长串数据分类为[[验证集自变量, 验证集因变量, 测试集自变量, 测试集因变量]]
    
    save_need(need_data)