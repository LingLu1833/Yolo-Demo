"""
根据模板图片生成YOLO数据集
"""

from dataset_generator import generate_dataset


def main():
    print("=" * 50)
    print("YOLO Dataset Generator")
    print("=" * 50)
    generate_dataset()


if __name__ == "__main__":
    main()