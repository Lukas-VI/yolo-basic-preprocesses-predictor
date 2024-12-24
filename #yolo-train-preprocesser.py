# YOLO 训练预处理脚本
#这里用来裁剪并预处理对象分类(cls)模型所需的数据集

import os
import cv2

def read_label_file(label_path):
    """从标记文件中读取标签."""
    with open(label_path, 'r') as file:
        return [line.strip().split() for line in file.readlines()]

def write_label_file(label_path, boxes):
    """将新的标签写入文件."""
    with open(label_path, 'w') as file:
        for box in boxes:
            file.write(' '.join(map(str, box)) + '\n')

def crop_and_process_images(image_folder, label_folder, output_folder, process_option=None):
    """裁剪图像并处理，同时生成新的标记文件. """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图片文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg')):  # 支持多种图片格式
            image_path = os.path.join(image_folder, filename)
            print(f"正在读取图片: {image_path}")

            # 检查图片文件是否存在
            if not os.path.exists(image_path):
                print(f"错误: 文件不存在 {image_path}, 跳过...")
                continue
            
            # 对应的标记文件路径
            label_path = os.path.join(label_folder, f'{os.path.splitext(filename)[0]}.txt')

            # 检查标记文件是否存在
            if os.path.exists(label_path):
                # 读取图片
                image = cv2.imread(image_path)
                if image is None:
                    print(f"错误: 无法读取图片 {image_path}, 跳过...")
                    continue

                height, width, _ = image.shape
                labels = read_label_file(label_path)  # 读取标记信息
                new_labels = []

                for index, label in enumerate(labels):
                    class_id = int(label[0])
                    x_center = float(label[1]) * width
                    y_center = float(label[2]) * height
                    bbox_width = float(label[3]) * width
                    bbox_height = float(label[4]) * height

                    # 计算裁剪区域的坐标
                    x1 = max(0, int(x_center - bbox_width / 2))
                    y1 = max(0, int(y_center - bbox_height / 2))
                    x2 = min(width, int(x_center + bbox_width / 2))
                    y2 = min(height, int(y_center + bbox_height / 2))

                    # 确保裁剪区域有效
                    if x1 < x2 and y1 < y2:
                        # 裁剪图片
                        cropped_image = image[y1:y2, x1:x2]

                        # 处理图片
                        if process_option == 'canny':
                            cropped_image = cv2.Canny(cropped_image, 100, 200)

                        # 构造新的图片保存路径
                        new_image_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_new.png')

                        # 保存裁剪后的图片
                        if cropped_image.size > 0:
                            cv2.imwrite(new_image_path, cropped_image)
                        else:
                            print(f"警告: 裁剪后的图片为空 {new_image_path}, 跳过保存...")

                        # 更新新的标记为裁剪后的整个图片,直接舍弃原来的位置信息
                        new_label = [
                            class_id,
                        ]
                        new_labels.append(new_label)
                    else:
                        print(f"警告: 无效的裁剪区域 {filename}, 跳过...")

                # 将新的标签写入文件
                new_label_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_new.txt')
                write_label_file(new_label_path, new_labels)

# 示例调用
if __name__ == "__main__":
    image_folder = 'd:/dn/MBset/Img'  # 包含图片的文件夹路径
    label_folder = 'd:/dn/MBset/Img'  # 包含标记文件的文件夹路径
    output_dir = 'd:/dn/MBset/Img/C2baseline'  # 输出文件夹路径
    
    crop_and_process_images(image_folder, label_folder, output_dir, process_option=None)
