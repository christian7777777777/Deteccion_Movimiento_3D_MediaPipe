import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Inicializa Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Inicializa la cámara
cap = cv2.VideoCapture(2)

# Configurar la figura de Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # Modo interactivo para actualizar la gráfica en tiempo real

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Limpiar la figura
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')  # Cambio de ejes para corregir la orientación
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Detección de Postura Corporal en 3D')

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        points = np.array([[-lm.x, lm.z, -lm.y] for lm in landmarks])  # Ajuste en los ejes
        
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
    
    plt.draw()
    plt.pause(0.001)

    # Muestra el video en ventana
    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Libera los recursos
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
