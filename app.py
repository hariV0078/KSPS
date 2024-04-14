from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from yolov8 import YOLOv8Detector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

detector = YOLOv8Detector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(f'uploads/{filename}')
        return redirect(url_for('detect_objects', filename=filename))

@app.route('/detect_objects/<filename>')
def detect_objects(filename):
    video_path = f'uploads/{filename}'
    output_path = f'static/output_{filename}'
    detector.detect_objects_in_video(video_path, output_path)
    return render_template('index.html', output_video=output_path)

if __name__ == '__main__':
    app.run(debug=True)
