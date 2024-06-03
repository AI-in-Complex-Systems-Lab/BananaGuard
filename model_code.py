import torch
import cv2
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class ObjectDetectionModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the model architecture
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=91)
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Load state dict with strict=False to allow partial loading
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading state_dict: {unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transformation
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((320, 320)),  # Reduce the size for faster processing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_frame(self, frame):
        frame_tensor = self.preprocess(frame)
        return frame_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

    def predict(self, frame_tensor):
        with torch.no_grad():
            detection_results = self.model(frame_tensor)[0]
        return detection_results

    def visualize_detections(self, frame, detection_results, threshold=0.5):
        boxes, labels, scores = detection_results['boxes'].cpu(), detection_results['labels'].cpu(), detection_results['scores'].cpu()
        for box, score in zip(boxes, scores):
            if score > threshold:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame
