import cv2
from flask import Flask, request, render_template, send_from_directory
from Modules import PoseModule as pm
from Modules import HandTrackingModule as HTM
import os
import uuid
import time
import tempfile

app = Flask(__name__, static_url_path='/static')


current_directory = os.path.dirname(os.path.realpath(__file__))
static_folder = os.path.join(current_directory, 'static')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hand_estimation')
def hand_estimation():
    return render_template('hand_estimation.html')

@app.route('/pose_estimation')
def pose_estimation():
    return render_template('pose_estimation.html')


def generate_unique_key():
    timestamp = int(time.time())
    random_str = str(uuid.uuid4().hex)[:6]
    return f"{timestamp}_{random_str}"


def PoseEstimation(video_file, output_path):
    video_file.seek(0)

    temp_video_path = os.path.join(tempfile.gettempdir(), f"{generate_unique_key()}.MOV")
    video_file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    detector = pm.poseDetector()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

    while True:
        success, img = cap.read()
        if success:
            img_copy = img.copy()
            img_copy = detector.findPos(img_copy)
            lmList, img_copy = detector.getPosition(img_copy, draw_path=False)
            result.write(img_copy)
        else:
            break

    result.release()

    os.remove(temp_video_path)



def hand_tracking(video_file, output_path):
    video_file.seek(0)

    temp_video_path = os.path.join(tempfile.gettempdir(), f"hands_{generate_unique_key()}.MOV")
    video_file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    detector = HTM.handDetector()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)


    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), 10, size)

    while True:
        success, img = cap.read()
        if success:
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            result.write(img)
        else:
            break

    result.release()

    os.remove(temp_video_path)


@app.route('/process_video', methods=['POST'])
def process_video():
    video_file = request.files['video']
    unique_key = generate_unique_key()
    file_name = f'{unique_key}.mp4'
    output_path = f'{static_folder}/{unique_key}.mp4'
    print("Output Path:", output_path)
    PoseEstimation(video_file, output_path=output_path)
    return render_template('results.html', output_path=file_name)


@app.route('/process_video_hands', methods=['POST'])
def process_video_hands():
    video_file = request.files['video']
    unique_key = generate_unique_key()
    file_name_hands = f'{unique_key}.mp4'
    output_path = f'{static_folder}/{unique_key}.mp4'
    hand_tracking(video_file, output_path=output_path)
    return render_template('results_hand_tracking.html', output_path_hands=file_name_hands)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)