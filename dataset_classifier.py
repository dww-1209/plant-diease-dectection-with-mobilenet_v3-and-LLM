# 将比赛数据按照类别进行文件夹归类
# 2018.09.16 add
############################################
import os
import json
from tqdm import tqdm
import shutil


class ClassifyAsLabel:
    def __init__(self):
        """"""
        print("按照标签类别进行图像归类")

    def read_json(self,json_path):
        """读取json文件"""
        with open(json_path, 'r') as f:
            temp = json.loads(f.read())
            print("json nums is:{}".format(len(temp)))

        return temp

    def classify(self,img_path,json_path,out_path):
        """按照类别进行归类"""
        #读入json文件
        json_infos=self.read_json(json_path)

        #遍历获取标注文件中的文件名及对应的labels
        img_ids=[]
        img_labels=[]
        for info in tqdm(json_infos):
            img_ids.append(info['image_id'])
            img_labels.append(info['disease_class'])

        # 根据当前的标签类别，将图像移动到对应的类别目录下
        for name,value in tqdm(zip(img_ids,img_labels)):
            tmp=os.path.join(img_path,name)
            label_path=os.path.join(out_path,str(value))
            if not os.path.exists(label_path):
                os.mkdir(label_path)

            shutil.move(tmp,label_path)


if __name__=="__main__":
    demo=ClassifyAsLabel()
    #demo.classify("AgriculturalDisease_trainingset/images", "AgriculturalDisease_trainingset\AgriculturalDisease_train_annotations.json",
    #              "input/train")
    #demo.classify("AgriculturalDisease_validationset/images", "AgriculturalDisease_validationset\AgriculturalDisease_validation_annotations.json",
    #              "input/val")
    
