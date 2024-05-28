from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import base64
import io

# Initialize the Flask application
app = Flask(__name__)

# Function to get HSV color limits for the given BGR color
def get_limits(color):
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
    lower_limit = np.array([hsv_color[0] - 10, 100, 100])
    upper_limit = np.array([hsv_color[0] + 10, 255, 255])
    return lower_limit, upper_limit

# Function to detect and label a specific color in the image
def detect_and_label_color(img, color_name, bgr_color):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Get the lower and upper HSV limits for the color
    lower_limit, upper_limit = get_limits(color=bgr_color)
    # Create a mask for the color
    mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
    # Convert mask to PIL image to get bounding box
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()
    if bbox:
        x1, y1, x2, y2 = bbox
        # Draw a rectangle around the detected color
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2)
        # Add the color name label above the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, color_name, (x1, y1 - 10), font, 0.9, bgr_color, 2, cv2.LINE_AA)
    return img, mask

# Function to process the frame and detect multiple colors
def process_frame(frame):
    # Define the BGR colors for detection
    yellow = [0, 255, 255]
    blue = [255, 0, 0]
    red = [0, 0, 255]
    green = [0, 255, 0]
    cyan = [255, 255, 0]
    magenta = [255, 0, 255]
    orange = [0, 165, 255]
    purple = [128, 0, 128]
    brown = [42, 42, 165]
    pink = [203, 192, 255]

    # Detect and label each color in the frame
    img, _ = detect_and_label_color(frame, "Yellow", yellow)
    img, _ = detect_and_label_color(img, "Blue", blue)
    img, _ = detect_and_label_color(img, "Red", red)
    img, _ = detect_and_label_color(img, "Green", green)
    img, _ = detect_and_label_color(img, "Cyan", cyan)
    img, _ = detect_and_label_color(img, "Magenta", magenta)
    img, _ = detect_and_label_color(img, "Orange", orange)
    img, _ = detect_and_label_color(img, "Purple", purple)
    img, _ = detect_and_label_color(img, "Brown", brown)
    img, _ = detect_and_label_color(img, "Pink", pink)

    return img

# Function to generate frames from the webcam feed
def generate_frames():
    global frame_to_save
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        frame_to_save = processed_frame.copy()
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        # Yield the frame as part of the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to provide the video feed to the webpage
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to capture and save the current frame
@app.route('/capture', methods=['POST'])
def capture():
    _, buffer = cv2.imencode('.jpg', frame_to_save)
    img_str = base64.b64encode(buffer).decode('utf-8')
    with open("captured_image.jpg", "wb") as f:
        f.write(base64.b64decode(img_str))
    return jsonify({'image': 'data:image/jpeg;base64,' + img_str})

# Route to download the captured image
@app.route('/download')
def download():
    return send_file("captured_image.jpg", as_attachment=True)

# Main entry point to start the Flask application
if __name__ == '__main__':
    frame_to_save = None
    app.run(debug=True)
