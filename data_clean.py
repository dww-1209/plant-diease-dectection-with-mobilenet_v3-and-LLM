#coding=utf-8
###############################################
# 训练数据去重，删除标签模糊的训练数据
# 2018.10.11 add
###############################################
import os
from tqdm import tqdm
from collections import Counter

def load_files(file_path):
    """
    载入文件
    :param file_path: 图像路径
    :return 返回文件名list，文件名及其路径的dict
    """
    filename_lists=list()
    filepath_dicts=dict()
    for directory,folder,filenames in os.walk(file_path):
        for filename in tqdm(filenames):
            if os.path.splitext(filename)[-1] not in [".jpg",'.JPG','.JPEG','.jpeg','.png']:
                continue

            filename_lists.append(filename)
            filepath_dicts[filename]=os.path.join(directory,filename)

    return filename_lists,filepath_dicts

def derepeat(name_list):
    """
    删除重复的图像名
    :param name_list: 图像名列表
    :param path_list: 图像名所在的路径列表
    :return:
    """
    new_name_list=list()
    for n in name_list:
        if "副本" in n:
            print("src n:{}".format(n))
            n=n.split("-")[0].strip(" ")+os.path.splitext(n)[-1]
            #print("n:{}".format(n))

        new_name_list.append(n)

    # 统计每张图片出现的次数,删除出现重复次数的图像名
    count=Counter(new_name_list)
    print("src nums:{}".format(len(new_name_list)))
    new_name_list=list(set(new_name_list))
    print("dst nums:{}".format(len(new_name_list)))
    for key,value in count.items():
        if value>1:
            new_name_list.remove(key)

    return new_name_list

def remove_file(name_list,path_list):
    """
    删除重复出现的图像
    :param name_list: 图像名
    :param path_list: 图像及对应路径的dict
    :return: 删除后的图像及路径名字典
    """
    new_dict=dict()
    delete_name_list=list()
    for k,v in path_list.items():
        if k not in name_list:
            # os.remove(v)
            delete_name_list.append(v)  #后面还需要对待副本字样的图像进行检查其是否存在相同的副本
        else:
            new_dict[k]=v

    # 对于有些带副本的图像不一定存在重复的图像
    tmp=set(name_list).difference(set(new_dict.keys()))
    print(tmp)
    for value in tmp:
         #print("value:{}".format(value))
         for k in path_list.keys():
             if os.path.splitext(value)[0] in k:
                 #print(path_list[k])
                 # 在操作前先判断，因为有些图像名可能是部分匹配
                 if path_list[k] in delete_name_list:
                     new_dict[k] = path_list[k]
                     delete_name_list.remove(path_list[k])
                     # print(k)
                     name_list[name_list.index(value)]=k

    # 删除图像文件
    for t in delete_name_list:
        os.remove(t)

    return new_dict,name_list

def dereplicate(train_file_path):
    """
    单个数据集(如训练集，测试集)去重复
    :param train_file_path:
    :return:
    """
    # 训练数据处理
    train_name_list,train_path_dict=load_files(train_file_path)
    print("train img nums:{},path nume:{}".format(len(train_name_list),len(train_path_dict)))
    train_name_list=derepeat(train_name_list)
    # print("num:{}".format(len(train_name_list)))
    train_path_dict,train_name_list=remove_file(train_name_list,train_path_dict)
    print("clean img nums:{},path nums:{}".format(len(train_name_list),len(train_path_dict)))

    return train_name_list,train_path_dict

def process_repeat(train_file_path,val_file_path):
    """
    查找训练集和测试集合中重复的图像名
    :param train_list:　训练集中的图像名列表
    :param val_list:　测试集中的图像名列表
    :return:　返回相同的图像名
    """
    # 训练集，验证集分别去重
    train_name_list, train_path_dict=dereplicate(train_file_path)
    val_name_list, val_path_dict=dereplicate(val_file_path)

    # 测试集和验证集取交集
    same=set(train_name_list)&set(val_name_list)
    print("验证集和测试集合的交集:{}".format(len(same)))

    # 删除对应的图像路径
    for v in same:
        if v in train_path_dict.keys():
            print("删除图像路径:{}".format(v))
            os.remove(train_path_dict[v])
            os.remove(val_path_dict[v])


if __name__=="__main__":
    process_repeat("AgriculturalDiease_trainingset","AgriculturalDiease_validationset")