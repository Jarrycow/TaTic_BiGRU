import random
import numpy as np
import pandas as pd
import Get_filename_1 as GF
from sklearn import model_selection
def main(file_dir,num_packs):  # 
    '''
    参数
    - file_dir: 文件目录
    - num_packs: 数据包数量
    返回:
    - all_data: [序列]
    '''
    dict1 = {}  # {APP: 文件名}
    classname = []  # [APPs]
    dir = GF.get_name(file_dir)  # 读取文件夹下所有文件名
    
    def read_data(dir):  # 读取每个文件的序列
        '''
        参数:
        - dir: [文件名]
        返回:
        - text: [序列]
        '''
        text = []  # 初始化一个空的列表来存储处理过的数据
        for name in dir:
            temp = pd.read_csv(name, header=None)  # 读取文件为pandas格式，[文件, [序列], APP编码]
            temp = temp.values  # 转为numpy格式
            temp_data = temp[:, 1:2]  # 序列
            idex = -1  # 序列下标
            for e in temp_data:  # 
                idex = idex + 1
                e1 = e[0].strip('[]').replace("'", "").replace(" ", "").split(',')  # 去除列表中的[ ] ,
                if len(e1) < 3 * num_packs:  # 零填充
                    e1 = e1 + [0, 0, 0] * ((3 * num_packs - len(e1)) // 3)
                else:
                    e1 = np.array(np.array(e1)[0:3 * num_packs], 'float32')
                temp1 = []
                temp1.append(temp[idex][-1])  # 将APP代码插在第一个
                for i in range(num_packs):
                    temp1.append(int(e1[3 * i]))
                text.append(temp1)
        return text


    for name in dir:  # 根据APP分类消息
        name1=name.split('/')[-2]
        if name1 not in dict1:
            classname.append(name1)
            dict1[name1]=[]
            dict1[name1].append(name)
        else:
            dict1[name1].append(name)

    
    def scramble_data(text):  # 数据打乱处理
        cc = list(zip(text))
        random.seed(100)
        random.shuffle(cc)
        text[:] = zip(*cc)
        return text[0]


    all_data=[]
    for name in classname:
        data = read_data(dict1[name])  # 获取该APP所有[序列]
        da = scramble_data(data)  # 打乱顺序
        if len(da)<=5000:  # 截取前5000份序列
            da=da
        else:
            da=da[:5000]
        all_data+=da
    return all_data
