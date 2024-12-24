#为提高yolov8的推理精度，我们可以对图片进行预处理，主要是为了提高准确率。
#推理的过程分为以下几个步骤：
#1. 对大图片进行一次推理，得到预测的位置信息结果
#2. 对预测的位置信息结果进行裁剪，得到小图块
#3. 对小图块进行预处理并进行一次推理，得到预测的类型信息结果
#4. 对预测的位置信息结果进行合并，得到最终的预测结果
#要载入两个模型，一次是大模型(BOX)，一次是小模型(CLS)，并且需要对大图片进行切割，对小图块进行推理，最后再进行合并。

#导入所需的库
import random
import cv2
import numpy as np
import time as t
from sympy import plot
from ultralytics import YOLO
 
# 设置参数
class Config:
    def __init__(self, model_path, img_path,subModel_path):
        self.model_path = model_path  # 主模型路径
        self.subModel_path = subModel_path  # 子模型路径
        self.img_path = img_path      # 输入图片路径
        self.model_size = 1280  # 主模型输入大小
        self.subModel_size = 640      # 子模型输入大小
    
    def display_config(self):
        print(f"主模型路径: {self.model_path}")
        print(f"子模型路径: {self.subModel_path}")
        print(f"输入图片路径: {self.img_path}")
        print(f"主模型输入大小: {self.model_size}")
        print(f"子模型输入大小: {self.subModel_size}")

# 定义模型，废

class Main_Model:
    def __init__(self, config, img):
        self.config = config
        self.img = img
        self.model_path = config.model_path

    def inference(self, img,model_path):
        # 进行一次推理
        ticks = str(t.asctime())
        # Load a model
        model = YOLO(model_path) 

        results = model.cpu()(img) 

        xyxy = []
        for result in results:
            #xyxy.append(result.boxes.xyxy)# Boxes object
            xyxy.extend(result.boxes.xyxy.numpy())
        print(xyxy)
        # 函数返回所有边界框
        return xyxy



class Processer:
    def __init__(self, config, img, xyxy):
        self.config = config
        self.img = img
        self.xyxy = xyxy

    # 根据得到预测的位置信息结果，把有对象的图片拆出来
    def cut_process_img(self, img):
        # 输入框列表
        images = []
        
        for box in self.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped_img = img[y1:y2, x1:x2] # numpy.ndarray
            images.append(cropped_img)
        
        return images
        
    def canny_process_img(self,img):
        # 进行canny边缘检测
        cannylist=[]

        '''for clip in img:
            gray = cv2.cvtColor(clip, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            cannylist.append(cv2.Canny(blur, 50, 150))
        '''

        for clip in img:

            gray = cv2.cvtColor(clip, cv2.COLOR_BGR2GRAY) if len(clip.shape) == 3 else clip
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            canny = cv2.Canny(blur, 50, 150)
            cannylist.append(canny)
            
            # 调试canny数组的形状
            print(f"Canny array shape: {canny.shape}")

        return cannylist


    '''cls_task的可用性没有得到证明！！！'''
    def cls_task(self, cannylist):
        """处理,生成结果列表. """
        
        # 创建输出图片列表
        cls = []
        # Load a submodel
        model = YOLO(config.subModel_path) 

        # 遍历图片列表中的所有图片
        for img in cannylist:
            # 读取图片，格式为带有 BGR 频道的 HWC 格式 uint8 (0-255)的 numpy.ndarray
            # 转为 RGB 格式的 HWC 格式 uint8 (0-255)的 numpy.ndarray
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #调试，保存小图片
            cv2.imwrite(str(t.time())+'.jpg',img)
            
            
            results = model.cpu()(img)
            for result in results:
                probs = result.probs
                print("probs:", probs)
                if probs != None:
                    #取最大概率的类别
                    cls.append(probs)
                else:
                    cls.append('Error')
        
        return cls

#画一个框
def plot_one_box(xyxy, im0, label=None, color=None, line_thickness=None):
    tl = line_thickness or round(0.002 * max(im0.shape[0:2])) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))#随机选颜色
    cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im0, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im0

# 主函数
if __name__ == '__main__':
    # 实例化配置类
    config = Config(
        model_path="D:/dn/PTs/Bmain.pt",
        subModel_path="D:/dn/PTs/BCsub.pt",
        img_path="D:/dn/111.jpg", 
    )
    
    config.display_config() 
    
    # 读取图片
    cvimg = cv2.imread(config.img_path)
    img = cvimg

    # 进行大图推理，获取boxes和classes
    locate_list = []
    type_list = []
    main_model = Main_Model(config, img)
    #推理函数返回了ReturnResult = zip(xyxy, classes)
    locate_list=main_model.inference(img,config.model_path)

    #对大图进行切割
    processer = Processer(config, img, locate_list)
    cut_images = processer.cut_process_img(img)
    
    #对小图进行预处理
    canny_images = processer.canny_process_img(cut_images)
    
    #对小图进行推理
    cls = processer.cls_task(canny_images)
    print("cls:", cls)

    #画框

    for i in range(len(cls)):#len(cls)):
        xyxy=locate_list[i]
        label=cls[i]
        img = plot_one_box(xyxy, img, label, color=None, line_thickness=None)
        cv2.imwrite(str(t.time())+'.jpg',img)

