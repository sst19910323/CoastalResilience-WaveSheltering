# filename: prototype_coastline_direction.py
# description: [PROTOTYPE] Early experiment using Sobel operators to calculate 
#              coastline orientation vectors from a test image.
#              This logic was successfully integrated into src/1_analyze_coastline.py.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 注意：此脚本依赖根目录下的 coast_test.png，如果该文件已删除，此脚本将无法运行
TEST_IMAGE = 'coast_test.png'

if not os.path.exists(TEST_IMAGE):
    print(f"Test image {TEST_IMAGE} not found. This is just a prototype archive.")
else:
    # 读取原始PNG图像
    image = cv2.imread(TEST_IMAGE, cv2.IMREAD_GRAYSCALE)

    # 将图像二值化
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # 使用更大卷积核的Sobel算子
    sobelx = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi  # 转换为角度

    # 找到边缘像素（海岸线）
    threshold = 50  # 可以根据需要调整阈值
    edges = magnitude > threshold

    # 创建一个空图像来标注方向
    direction_image = np.zeros_like(binary_image, dtype=np.float32)

    # 标注方向
    direction_image[edges] = direction[edges]

    # 可视化方向图像
    plt.figure(figsize=(10, 8))
    plt.imshow(direction_image, cmap='hsv')
    plt.colorbar(label='Angle (degrees)')
    plt.title('Prototype: Coastline Directions')
    plt.show()