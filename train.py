"""
训练YOLO模型
"""

from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    results = model.train(
        data="datasets/yolo_dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        save=True,
        save_period=10,
        project="datasets/runs",
        name="train",
        exist_ok=True,
    )

    print(f"Training complete. Results: {results}")


if __name__ == "__main__":
    main()