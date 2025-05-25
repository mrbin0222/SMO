import os

import cv2
import numpy as np
from ultralytics import YOLO


def predict_image(model_path, image_path, max_det=9999, iou=0.5, isbox=True, imgsz=1000):
        """
        predict on a single image
        
        Args:
            model_path (str): path to model weights
            image_path (str): path to image to predict
            max_det (int): maximum number of detected boxes, default 9999
            iou (float): IoU threshold, default 0.5
            isbox (bool): return format, True returns bounding box coordinates, False returns center point coordinates
            imgsz (int): image size, default 1000
        
        Returns:
            list: if isbox=True, return bounding box coordinates list [[x1,y1,x2,y2], ...]
                if isbox=False, return center point coordinates list, each element shape is (1,2)
        """
        try:
            # check if file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"model file not found: {model_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"image file not found: {image_path}")
                
            # load model
            model = YOLO(model_path)
            
            # predict
            results = model.predict(source=image_path, max_det=max_det, iou=iou, imgsz=imgsz)
            
            if len(results) == 0:
                return []
                
            # get detected boxes
            boxes_xyxy = []
            for box in results[0].boxes:
                # get [x1,y1,x2,y2] format coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes_xyxy.append([x1, y1, x2, y2])
                
            if not isbox:
                # calculate center point coordinates
                points = []
                for box in boxes_xyxy:
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    # each point is a (1,2) shape ndarray
                    point = np.array([[center_x, center_y]])
                    points.append(point)
                return points
            
            return boxes_xyxy
            
        except Exception as e:
            print(f"error during prediction: {str(e)}")
            return []



# # example usage
# if __name__ == "__main__":
#     # example parameters
#     model_path = "./runs/train/train/weights/best.pt"
#     image_path = "./MoNuSegTestData/test/images/test.png"
    
#     # get bounding box coordinates
#     boxes = predict_image(model_path, image_path, isbox=True)
#     print("detected bounding box coordinates:")
#     print(boxes)
    
#     # get center point coordinates
#     points = predict_image(model_path, image_path, isbox=False)
#     print("\ndetected center point coordinates:")
#     print(points)
#     print("shape of coordinate array:", points.shape)  # show overall shape
#     print("shape of first point:", points[0].shape)  # should show (1, 2)

