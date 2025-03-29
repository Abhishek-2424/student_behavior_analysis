from flask import Flask, render_template, request, send_from_directory, url_for
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Set environment to production
app.config['ENV'] = 'production'

# Define file paths
UPLOAD_FOLDER = 'data/images'
OUTPUT_FOLDER = 'output'

# Create output directories if not exist
if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'images')):
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'))

if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'videos')):
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'videos'))

# Load the trained YOLO model
model = YOLO('models/best(3).pt')  # Path to your trained YOLOv8 model

# Behavior categories (ensure these match the model's class labels)
# BEHAVIORS = ['Using_phone', 'bend', 'book', 'bow_head', 'hand-raising', 'phone', 
            #  'raise_head', 'reading', 'sleep', 'turn_head', 'upright', 'writing']

BEHAVIORS = ['laptop', 'laughing', 'looking away', 'mobile phone', 'reading', 'sleeping', 'using laptop', 'writing']

# Route for handling form submission
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process image/video
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Check if the file is an image or a video
    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        output_video_path = os.path.join(OUTPUT_FOLDER, 'videos', file.filename)
        process_video(file_path, output_video_path)
        # Generate video URL for frontend
        output_video_url = url_for('static_file', folder='videos', filename=file.filename)
        return render_template('index.html', video_url=output_video_url, download_url=output_video_url, filename=file.filename)
    elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        output_image_path = os.path.join(OUTPUT_FOLDER, 'images', file.filename)
        behaviors, output_image_url = process_image(file_path, output_image_path)
        # Generate image URL for frontend
        return render_template('index.html', image_url=output_image_url, download_url=output_image_url, filename=file.filename, behaviors=behaviors)
    else:
        return 'Unsupported file type', 400

# Process image with YOLO and count behaviors
def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    results = model(img)  # Perform inference

    # Get class names and their corresponding counts
    behavior_counts = {behavior: 0 for behavior in BEHAVIORS}

    # For each detection, check the class and update the count
    for result in results:
        for detection in result.boxes:
            class_id = int(detection.cls[0].item())
            if 0 <= class_id < len(BEHAVIORS):
                behavior = BEHAVIORS[class_id]
                behavior_counts[behavior] += 1

    # Plot the bounding boxes on the image
    output_image = results[0].plot()
    cv2.imwrite(output_path, output_image)

    # Return behavior counts and output image URL
    output_image_url = url_for('static_file', folder='images', filename=os.path.basename(output_path))
    return behavior_counts, output_image_url

# Process video with YOLO
def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame_output = results[0].plot()  # Get the first result and plot on the frame
        out.write(frame_output)

    cap.release()
    out.release()

# Serve the processed static files
@app.route('/static/<folder>/<filename>')
def static_file(folder, filename):
    directory = os.path.join(OUTPUT_FOLDER, folder)
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, port=5001)
