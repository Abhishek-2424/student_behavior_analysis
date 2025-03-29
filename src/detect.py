import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('models/best(3).pt')

# For image
image_path = 'data/images/classroom.jpg'
img = cv2.imread(image_path)
results = model(img)
results.save('output/images/test_image_output.jpg')  # Save the image with detections

# For video
video_path = 'data/images/test_video.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/images/test_video_output.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    frame_output = results.render()[0]
    out.write(frame_output)

cap.release()
out.release()
