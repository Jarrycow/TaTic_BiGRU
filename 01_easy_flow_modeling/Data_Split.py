import random
import numpy as np
import pandas as pd
import Get_filename as GF


def main(file_dir,num_packs1,num_packs2=32):
    '''
    输入：
        file_dir: 数据所在文件夹
        num_packs1
        num_packs2=32
    输出：

    '''
    dict1 = {}
    classname = []  # APP列表
    dirs = GF.get_name(file_dir)  # 获取目录下所有文件名[列表]
    def read_data(dirs,num_packs):
        '''
        读取数据
        输入: 
            dirs: [某APP下所有数据]
            num_packs: int(读取数据包个数)
        输出: 
        '''
        text = []
        for name in dirs:  # 迭代该APP下所有数据
            temp = pd.read_csv(name, header=None)  # 读取文件
            temp = temp.values  # 转换为 array
            temp_data = temp[:, 1:2]  # temp_data为[数据流]
            idex = -1  
            for e in temp_data: # 迭代数据流
                idex = idex + 1 
                e1 = e[0].strip('[]').replace("'", "").replace(" ", "").split(',')
                if len(e1) < 3 * num_packs:  # 数据包个数不够，补0
                    e1 = e1 + [0] * (3 * num_packs - len(e1))
                text.append([temp[idex][0], np.array(np.array(e1)[0:3 * num_packs], 'float32'), temp[idex][-1]])
        return text


    for name in dirs:  # 迭代所有文件，classname [生成APP]，dict1 {APP: APP所有路径}
        print("data name\t" + str(dirs.index(name)) + '\t' + str(len(dirs)))
        name1=name.split('/')[-2]  # 获取该文件所属APP
        if name1 not in dict1:  # 
            classname.append(name1)
            dict1[name1]=[]
            dict1[name1].append(name)
        else:
            dict1[name1].append(name)

    def scramble_data(text):
        cc = list(zip(text))  # 打包成元组
        random.seed(100)
        random.shuffle(cc)  # 打乱顺序
        text[:] = zip(*cc)
        return text[0]


    short_all_data=[]
    long_all_data=[]
    for name in classname:  # 迭代所有APP
        print("data classname\t" + str(classname.index(name)) + '\t' + str(len(classname)))
        short_data = read_data(dict1[name],num_packs1)  # 短包数据
        long_data = read_data(dict1[name], num_packs2)  # 长包数据
        short_da = scramble_data(short_data)
        short_da=short_da[:5000]
        short_all_data+=short_da

        long_da = scramble_data(long_data)
        long_da=long_da[:5000]
        long_all_data+=long_da
    return short_all_data,long_all_data
