"""
检测YOLO数据集中的图片与标注是否准确
"""

import os
import cv2

IMAGES_DIR = os.path.join("datasets", "yolo_dataset", "train", "images")
LABELS_DIR = os.path.join("datasets", "yolo_dataset", "train", "labels")


def parse_yolo_label(label_path, img_width, img_height):

    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x1 = int((cx - w / 2) * img_width)
            y1 = int((cy - h / 2) * img_height)
            x2 = int((cx + w / 2) * img_width)
            y2 = int((cy + h / 2) * img_height)

            boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def draw_boxes(img, boxes):
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for class_id, x1, y1, x2, y2 in boxes:
        color = colors[class_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"class {class_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )


def main():
    if not os.path.isdir(IMAGES_DIR):
        print(f"[ERROR] 图片目录不存在: {os.path.abspath(IMAGES_DIR)}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        [
            f
            for f in os.listdir(IMAGES_DIR)
            if os.path.splitext(f)[1].lower() in extensions
        ]
    )

    if not image_files:
        print(f"[INFO] 图片目录中没有找到图片: {os.path.abspath(IMAGES_DIR)}")
        return

    total = len(image_files)
    print(f"共找到 {total} 张图片")
    print("按 任意键 查看下一张，按 ESC 退出\n")

    window_name = "YOLO Dataset Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(IMAGES_DIR, image_file)
        stem = os.path.splitext(image_file)[0]
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] 无法读取图片: {image_file}，跳过")
            continue

        img_height, img_width = img.shape[:2]

        boxes = parse_yolo_label(label_path, img_width, img_height)

        draw_boxes(img, boxes)

        info_text = f"[{idx + 1}/{total}] {image_file}  |  boxes: {len(boxes)}"
        cv2.putText(
            img,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, img)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            print("ESC 按下，退出程序")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
