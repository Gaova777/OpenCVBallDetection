import cv2 as cv
import numpy as np

# Función para calcular la distancia euclidiana
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Cálculo de velocidad
def calculate_speed(prev_positions, prev_time, pixel_to_meter):
    if len(prev_positions) >= 2:
        current_time = cv.getTickCount() / cv.getTickFrequency()
        if prev_time is not None:
            displacement = np.linalg.norm(np.array(prev_positions[-1]) - np.array(prev_positions[-2]))
            time_sec = current_time - prev_time

            if time_sec > 0:
                return (displacement / time_sec) * pixel_to_meter, current_time

    return None, cv.getTickCount() / cv.getTickFrequency()

# Dibujar la detección y predicción
def draw_tracking_info(frame, ball_position, prediction, speed):
    if ball_position:
        cv.circle(frame, ball_position, 10, (0, 255, 0), 2)
        cv.circle(frame, ball_position, 2, (0, 0, 255), 3)

    cv.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 0, 0), -1)

    if speed is not None:
        cv.putText(frame, f"Velocidad: {speed:.2f} m/s", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
