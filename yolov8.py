import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

class YOLOv8Detector:
    def __init__(self, model_path='C:/Users/LENOVO/yolov8s.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        # Load your YOLOv8 model here
        model_state_dict = torch.load(model_path, map_location=self.device)
        model = self.build_model()  # Assuming build_model() creates the model architecture
        model.load_state_dict(model_state_dict)
        return model

    def build_model(self):
        # Define your YOLOv8 model architecture here
        # Example:
        model = ...  # Define your model
        return model

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def detect_objects_in_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_tensor = self.preprocess_image(image)
                
                # Forward pass through YOLOv8 model
                outputs = self.model(image_tensor)
                
                # Post-process outputs to get bounding boxes
                detected_boxes = self.post_process(outputs)

                # Draw bounding boxes on the frame
                for box in detected_boxes:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, box[4], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                out.write(frame)

        cap.release()
        out.release()

    def post_process(self, outputs, confidence_threshold=0.5):
        detected_boxes = []
        for output in outputs:
            # Extract class predictions, box coordinates, and confidence scores
            class_ids = output[..., 5:].argmax(dim=-1)
            confidences = torch.sigmoid(output[..., 4])  # Assuming confidence score is in the 5th index
            confidences *= output[..., 5:].max(dim=-1)[0]  # Multiply class confidence by objectness score
            box_coords = output[..., :4]

            # Filter out low-confidence detections
            mask = confidences > confidence_threshold
            class_ids = class_ids[mask]
            confidences = confidences[mask]
            box_coords = box_coords[mask]

            # Convert box coordinates from YOLO format to (x1, y1, x2, y2) format
            box_coords = self.yolo_to_xyxy(box_coords)

            # Iterate over each detected box
            for i in range(len(class_ids)):
                class_id = class_ids[i].item()
                confidence = confidences[i].item()
                box = box_coords[i].tolist()

                # Map class IDs to class names
                class_names = ['pedestrians', 'trucks', 'cars', 'auto', 'bikes']
                class_name = class_names[class_id]

                # Add the detected box to the list of detected boxes
                detected_boxes.append([int(coord) for coord in box] + [class_name])

        return detected_boxes

    def yolo_to_xyxy(self, boxes):
        box_xy = boxes[..., :2]
        box_wh = boxes[..., 2:4]
        box_min = box_xy - (box_wh / 2.0)
        box_max = box_xy + (box_wh / 2.0)
        return torch.cat([box_min, box_max], dim=-1)
