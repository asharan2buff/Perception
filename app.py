import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Title of the app
st.title("YOLOv8s vs Fused YOLOv8s Object Detection Comparison")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Class names
class_names = ['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck', 'Misc', 'Tram', 'Person_sitting', 'DontCare']

# Load models (assuming files are in the same directory or adjust paths)
yolo_model = YOLO("yolov8s.pt")
fused_model = YOLO("best.pt")

# Function to draw boxes on image
def draw_boxes(image, boxes, is_prediction=False, scores=None, is_fused=True):
    img_copy = image.copy()
    if len(boxes) == 0:
        return img_copy
    for i, box in enumerate(boxes):
        try:
            if is_prediction:
                if len(box) < 5:  # Ensure box has class index
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                class_idx = int(box[4])
                class_name = class_names[class_idx] if 0 <= class_idx < len(class_names) else 'Unknown'
                score = scores[i] if scores is not None and i < len(scores) else 1.0
                label = f"{class_name} {score:.2f}"
                color = (255, 0, 0) if is_fused else (0, 0, 255)  # Red for fused, Blue for non-fused
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_copy, label, (x1, max(y1-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except (IndexError, KeyError) as e:
            st.write(f"Error drawing box {i}: {e}, skipping")
    return img_copy

# Process image and display results
if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    rgb_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Predict with YOLO model
    yolo_results = yolo_model.predict(source=rgb_img, imgsz=640, conf=0.25)
    if yolo_results[0].boxes.xyxy.numel() == 0:
        st.write("Warning: No YOLO predictions")
        yolo_boxes = np.array([])
        yolo_scores = np.array([])
        yolo_classes = np.array([])
    else:
        yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        yolo_scores = yolo_results[0].boxes.conf.cpu().numpy()
        yolo_classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
    yolo_boxes_with_cls = np.column_stack((yolo_boxes, yolo_classes))
    yolo_pred_img = draw_boxes(rgb_img, yolo_boxes_with_cls, is_prediction=True, scores=yolo_scores, is_fused=False)

    # Predict with fused model
    fused_results = fused_model.predict(source=rgb_img, imgsz=640, conf=0.25)
    if fused_results[0].boxes.xyxy.numel() == 0:
        st.write("Warning: No fused predictions")
        fused_boxes = np.array([])
        fused_scores = np.array([])
        fused_classes = np.array([])
    else:
        fused_boxes = fused_results[0].boxes.xyxy.cpu().numpy()
        fused_scores = fused_results[0].boxes.conf.cpu().numpy()
        fused_classes = fused_results[0].boxes.cls.cpu().numpy().astype(int)
    fused_boxes_with_cls = np.column_stack((fused_boxes, fused_classes))
    fused_pred_img = draw_boxes(rgb_img, fused_boxes_with_cls, is_prediction=True, scores=fused_scores, is_fused=True)

    # Create a figure for visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(rgb_img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    ax2.imshow(yolo_pred_img)
    ax2.set_title("YOLOv8s Predictions")
    ax2.axis('off')
    ax3.imshow(fused_pred_img)
    ax3.set_title("Fused YOLOv8s Predictions")
    ax3.axis('off')

    # Display the plot in Streamlit
    st.pyplot(fig)
else:
    st.write("Please upload an image to proceed with object detection.")