import os
import cv2
import uuid
import base64
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
from PIL import Image
import time

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/frames/<folder>/<filename>')
def frame_file(folder, filename):
    return send_from_directory(os.path.join(FRAMES_FOLDER, folder), filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'video': {
                'path': filepath,
                'url': f"/uploads/{filename}",
                'filename': filename
            }
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/extract-frames', methods=['POST'])
def extract_frames():
    data = request.json
    video_path = data.get('videoPath')
    
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400
    
    try:
        # Create a unique folder for frames
        video_filename = os.path.basename(video_path)
        base_filename = os.path.splitext(video_filename)[0]
        frames_dir = os.path.join(FRAMES_FOLDER, base_filename)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video properties using OpenCV
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        # Calculate how many frames to extract (max 10 for testing)
        max_frames = 10
        frame_interval = max(1, frame_count // max_frames)
        
        frames = []
        count = 0
        frame_index = 0
        
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_index % frame_interval == 0:
                frame_filename = f"frame_{count}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                frames.append({
                    'path': f"{FRAMES_FOLDER}/{base_filename}/{frame_filename}",
                    'url': f"/frames/{base_filename}/{frame_filename}"
                })
                count += 1
            
            frame_index += 1
        
        cap.release()
        
        return jsonify({
            'success': True,
            'frames': frames,
            'videoInfo': {
                'duration': duration,
                'framesDir': base_filename
            }
        })
    
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return jsonify({'error': 'Failed to extract frames', 'details': str(e)}), 500

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    data = request.json
    frame_path = data.get('framePath')
    
    if not frame_path:
        return jsonify({'error': 'No frame path provided'}), 400
    
    try:
        # Open the image using PIL
        image = Image.open(frame_path)
        
        # Initialize Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create a prompt
        prompt = "Provide a concise description (1-2 sentences) of what's shown in this video frame."
        
        # Get response from Gemini
        response = model.generate_content([prompt, image])
        
        # Extract and clean the description
        description = response.text.strip()
        
        return jsonify({
            'success': True,
            'description': description
        })
    
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return jsonify({'error': 'Failed to analyze frame', 'details': str(e)}), 500

@app.route('/api/trim-video', methods=['POST'])
def trim_video():
    data = request.json
    video_path = data.get('videoPath')
    start_time = data.get('startTime')
    end_time = data.get('endTime')
    
    if None in [video_path, start_time, end_time]:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Generate output filename
        video_ext = os.path.splitext(video_path)[1]
        output_filename = f"trimmed_{int(time.time())}{video_ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Trim video using MoviePy
        with VideoFileClip(video_path) as video:
            # Extract the subclip
            new_clip = video.subclip(start_time, end_time)
            # Write the result to a file
            new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        return jsonify({
            'success': True,
            'message': 'Video trimmed successfully',
            'video': {
                'path': output_path,
                'url': f"/output/{output_filename}"
            }
        })
        
    except Exception as e:
        print(f"Error trimming video: {e}")
        return jsonify({'error': 'Failed to trim video', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)