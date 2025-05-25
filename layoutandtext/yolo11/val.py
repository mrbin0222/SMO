from ultralytics import YOLO

# Load a model
model = YOLO("./runs/train/train1/weights/best.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # map50-95