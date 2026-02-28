import os
import torch
from PIL import Image,ImageTk
from torchvision import transforms ,models
import tkinter as Tk
from tkinter import filedialog
import numpy as np
from torch import nn
from read_txt import read_txt_to_dict
import matplotlib.pyplot as plt
import cv2
 
#import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

global img_path

class_indict = read_txt_to_dict(file_path='actual_classed_v2.txt')

def main(img_path):
    IMG_SIZE = (224, 224)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # for MobileNetV2
    ])



    # load image
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    #plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # # read class_indict
    # json_path = './class_indices_15cls.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # create model
    #model = mobile_net_v2(num_classes=15)
    model = models.mobilenet_v2()

    backbone_layers = list(model.features.children()) 
    head_layers = model.classifier

    # 3) Unfreeze top 30% of the backbone (excluding BatchNorm layers for stability)
    UNFREEZE_RATIO = 0.30
    start_unfreeze = int(len(backbone_layers) * (1 - UNFREEZE_RATIO))

    # Freeze layers in the backbone and unfreeze top 30%
    for i, layer in enumerate(backbone_layers):
        if i >= start_unfreeze and not isinstance(layer, nn.BatchNorm2d):
            for param in layer.parameters():
                param.requires_grad = True  # Unfreeze this layer
        else:
            for param in layer.parameters():
                param.requires_grad = False  # Freeze this layer


    # 4) Modify the classifier for fine-tuning
    num_classes = 61  # Replace with your actual number of classes
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),  # Regularization
        nn.Linear(model.last_channel, num_classes)  # Output layer with 'num_classes' classes
    )

    model.to(device='cuda:0')
    # load model weights
    weights_path = 'dual_stream_mobilenet_best.pth'
    # weights_path = "mobilenetv2_best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device,weights_only=True),strict=False)

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = int(torch.argmax(predict).numpy())

    print_res = "class: {}   prob: {:.3}".format(predict_cla,
                                                 predict[predict_cla].numpy())
    #plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(i,
                                                  predict[i].numpy()))
    # 获取最大概率及其索引
    max_prob_index = np.argmax(predict.numpy())
    max_prob = predict[max_prob_index]
    Plant_class = class_indict[max_prob_index][0]
    Healthy = class_indict[max_prob_index][1]
    Diseased_degree = class_indict[max_prob_index][2]
    Diseased_class = class_indict[max_prob_index][3]
    # 更新打印结果
    print_res = "植物名称：{}   是否患病：{}    \n 病害名称: {}    患病程度：{}   相似度: {:.3f}".format(Plant_class,Healthy,Diseased_class,Diseased_degree, max_prob.item())
    print(print_res)
    text_label.config(text=print_res)

    #plt.show()


def select_image():

    filepath = filedialog.askopenfilename(
    initialdir=".",  # 初始目录，默认当前目录
    title="选择文件",  # 标题
    filetypes=(("Text files", "*.jpg"), ("All Files", "*.*"))  # 可选择文件类型
    )
    if filepath:
        img_path = filepath
        img = Image.open(filepath)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)  # 调整图片大小
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk  # 防止垃圾回收
        main(img_path)

def App():

    root = Tk.Tk()
    root.title("虫见分晓————病虫害识别")
    root.geometry("1100x600+400-200")
    btn1 = Tk.Button(root,text= "选取农作物图片",width=20,height=3,command=lambda :select_image())
    btn1.place(x = 120, y = 450)

    font_style = ("Arial",16)
    global img_label
    img_label = Tk.Label(root)
    img_label.place(x = 120, y = 80)
    #img_label.pack(pady=20)

    global text_label
    text_label = Tk.Label(root,font=font_style)
    text_label.place(x = 510,y = 350)
    root.mainloop()



if __name__ == '__main__':
    App()

    test_image = cv2.imread('test.jpg')
    # 将BGR转换为RGB（OpenCV读取的是BGR格式，matplotlib需要RGB格式）
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    true_label = 0

    true_Plant_class = class_indict[true_label][0]
    true_Healthy = class_indict[true_label][1]
    true_Diseased_degree = class_indict[true_label][2]
    true_Diseased_class = class_indict[true_label][3]


    fig, (ax_image, ax_text) = plt.subplots(1, 2, figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    ax_image.imshow(test_image)
    ax_image.axis('off')  

    # 设置右侧文本区域
    ax_text.set_facecolor('white')
    ax_text.axis('off')  

    # 在右边添加文本
    text_content = f"""
    - 测试图片真实情况如下：
    - 植物种类：{true_Plant_class}
    - 患病情况：{true_Healthy}
    - 患病种类：{true_Diseased_class}
    - 患病程度：{true_Diseased_degree}
    """

    # 添加文本到右边画布
    ax_text.text(0.05, 0.5, text_content, 
                transform=ax_text.transAxes,  # 使用相对坐标
                fontsize=26,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 调整布局
    plt.tight_layout()
    plt.show()

