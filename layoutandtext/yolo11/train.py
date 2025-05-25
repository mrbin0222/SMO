from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
model = YOLO("yolo11n.pt")
model = YOLO("yolo11n.yaml").load("yolo11n.pt")


model.train(data="Path/to/data.yaml", 
            epochs=50, 
            imgsz=1000, 
            batch=12, 
            project="runs/train",
            optimizer="SGD")