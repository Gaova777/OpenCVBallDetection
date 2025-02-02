import cv2 as cv
import numpy as np
import os
import time

# Deshabilitar Qt para evitar errores en entornos sin GUI (como EC2)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Inicializar el filtro de Kalman
def initialize_kalman():
    kalman = cv.KalmanFilter(6, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0, 0.5, 0],  
                                        [0, 1, 0, 1, 0, 0.5],  
                                        [0, 0, 1, 0, 1, 0],    
                                        [0, 0, 0, 1, 0, 1],    
                                        [0, 0, 0, 0, 1, 0],    
                                        [0, 0, 0, 0, 0, 1]], np.float32)

    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.0005  
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.02  
    return kalman

# Función para detectar la pelota
def detect_ball(frame, hsv, lower_colors, upper_colors):
    mask = None

    for lower, upper in zip(lower_colors, upper_colors):
        color_mask = cv.inRange(hsv, lower, upper)
        mask = color_mask if mask is None else cv.bitwise_or(mask, color_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        valid_contours = [c for c in contours if cv.contourArea(c) > 200]
        if valid_contours:
            max_contour = max(valid_contours, key=cv.contourArea)
            (x, y), radius = cv.minEnclosingCircle(max_contour)

            if 5 < radius < 80:  # Filtrar objetos demasiado pequeños o grandes
                return (int(x), int(y)), mask

    return None, mask

# Función para calcular la distancia
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Procesamiento de video con detección y velocidad
def process_video(video_path, output_path="videos/output.mp4"):
    cap = cv.VideoCapture(video_path)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    kalman = initialize_kalman()

    lower_colors = [np.array([100, 150, 50]), np.array([10, 100, 100])]
    upper_colors = [np.array([130, 255, 255]), np.array([25, 255, 255])]

    prev_positions = []
    prev_time = None
    speed_measurements = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        ball_position, mask = detect_ball(frame, hsv, lower_colors, upper_colors)

        if ball_position:
            measurement = np.array([[np.float32(ball_position[0])], [np.float32(ball_position[1])]])
            kalman.correct(measurement)
            prev_positions.append(ball_position)
            if len(prev_positions) > 10:
                prev_positions.pop(0)

        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        if ball_position:
            cv.circle(frame, ball_position, 10, (0, 255, 0), 2)  # Pelota detectada
            cv.circle(frame, ball_position, 2, (0, 0, 255), 3)  # Centro de la pelota

        cv.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)  # Predicción Kalman

        for i in range(1, len(prev_positions)):
            cv.line(frame, prev_positions[i - 1], prev_positions[i], (0, 255, 255), 2)

        if len(prev_positions) >= 2:
            current_time = cv.getTickCount() / cv.getTickFrequency()
            if prev_time is not None:
                displacement = dist(prev_positions[-1], prev_positions[-2])
                time_sec = current_time - prev_time

                if time_sec > 0:
                    speed_m_s = (displacement / time_sec) * (1 / 100)
                    speed_measurements.append(speed_m_s)
                    if len(speed_measurements) > 5:
                        speed_measurements.pop(0)
                    avg_speed = np.mean(speed_measurements)

                    cv.putText(frame, f"Velocidad: {avg_speed:.2f} m/s", (20, 40),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            prev_time = current_time

        out.write(frame)

    cap.release()
    out.release()
    return output_path
