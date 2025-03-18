import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Inicializa Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Configuración global para la captura de video
camera = None
camera_index = 0

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            # Intenta con la cámara predeterminada si la especificada no funciona
            camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Procesar el frame con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            # Dibujar landmarks en 2D en el frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Convertir el frame a formato JPEG para transmitir
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_3d_pose():
    camera = get_camera()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Procesar el frame con MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Limpiar la figura
        ax.cla()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('Detección de Postura Corporal en 3D')
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            points = np.array([[-lm.x, lm.z, -lm.y] for lm in landmarks])
            
            # Dibujar los landmarks
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o')
            
            # Dibujar conexiones
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = points[start_idx]
                end_point = points[end_idx]
                ax.plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        [start_point[2], end_point[2]], 'b')
        
        # Convertir la figura a una imagen JPEG
        canvas = FigureCanvas(fig)
        output = io.BytesIO()
        canvas.print_png(output)
        plot_data = base64.b64encode(output.getvalue()).decode('utf-8')
        
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + 
               output.getvalue() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_3d_feed')
def pose_3d_feed():
    return Response(generate_3d_pose(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_3d_plot')
def get_3d_plot():
    camera = get_camera()
    success, frame = camera.read()
    
    if not success:
        return jsonify({'error': 'No se pudo capturar el frame'})
    
    # Procesar el frame con MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Crear figura 3D
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Detección de Postura Corporal en 3D')
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        points = np.array([[-lm.x, lm.z, -lm.y] for lm in landmarks])
        
        # Dibujar los landmarks
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o')
        
        # Dibujar conexiones
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = points[start_idx]
            end_point = points[end_idx]
            ax.plot([start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]], 'b')
    
    # Convertir la figura a una imagen codificada en base64
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    plot_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return jsonify({'plot': plot_data})

@app.route('/change_camera/<int:index>')
def change_camera(index):
    global camera_index
    camera_index = index
    release_camera()  # Liberar la cámara actual
    return jsonify({'status': 'success', 'camera_index': camera_index})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        release_camera()