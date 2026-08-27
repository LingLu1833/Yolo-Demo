# 运行前请激活 conda 环境: conda activate yolo_env
# 运行命令: python main.py

import os

IMAGES_DIR = os.path.join("images")

# 模板目录路径，存放用于合成图像的模板文件
TEMPLATES_DIR = os.path.join("templates")

# 数据集输出目录，生成的 YOLO 格式数据集将保存在此目录下
OUTPUT_DIR = os.path.join("datasets", "yolo_dataset")

# 每个类别生成的图像数量
NUM_IMAGES_PER_CLASS = 2000

# 旋转角度范围（度），模板图像将被随机旋转此范围内的角度
ROTATION_RANGE = (-180, 180)

# 缩放比例范围，模板图像将被随机缩放至该范围内的比例 0.5是游戏内的比例
SCALE_RANGE = (0.5, 0.5)

# 输出图像尺寸 (宽, 高)，所有生成的图像将统一调整为此尺寸
IMAGE_SIZE = (640, 640)

# 每张图像中的目标数量范围，随机从此范围内选取目标数量
OBJECTS_PER_IMAGE = (1, 5)

# 最大重叠比例，当目标间重叠面积超过此比例时将重新放置
MAX_OVERLAP_RATIO = 0.4

# 训练集比例，数据集划分时训练集所占的比例
TRAIN_RATIO = 0.8

# 前景颜色增强开关，对模板前景施加 HSV / 亮度对比度变化，模拟不同光照条件
FOREGROUND_COLOR_AUGMENT = True

# 前景 HSV 偏移范围 (色相, 饱和度, 明度)
FOREGROUND_HSV_SHIFT = (0.05, 0.3, 0.3)

# 前景亮度对比度范围 (亮度因子, 对比度因子)，1.0 表示不变
FOREGROUND_BRIGHTNESS_RANGE = (0.7, 1.3)
FOREGROUND_CONTRAST_RANGE = (0.8, 1.3)

# 后续合成图像的增强开关，拉近与真实截图的分布距离
POST_PROCESS_AUGMENT = True

# 高斯模糊 sigma 范围，模拟运动模糊或对焦不准
POST_BLUR_SIGMA_RANGE = (0.0, 0.8)

# 高斯噪声标准差范围，模拟传感器噪声 / 压缩噪点
POST_NOISE_STD_RANGE = (0.0, 8.0)

# JPEG 压缩质量范围（越低噪点越多），模拟截图压缩
POST_JPEG_QUALITY_RANGE = (75, 100)

# 背景类型权重，控制各背景类型的生成概率
# 可选: "solid"(纯色), "linear_gradient"(线性渐变), "radial_gradient"(径向渐变),
#       "noise"(噪声纹理), "grid"(网格/条纹), "checker"(棋盘格), "from_image"(从background.png随机裁剪)
BACKGROUND_TYPE_WEIGHTS = {
    "solid": 0.1,
    "linear_gradient": 0.14,
    "radial_gradient": 0.1,
    "noise": 0.1,
    "grid": 0.14,
    "checker": 0.1,
    "from_image": 0.32,
}