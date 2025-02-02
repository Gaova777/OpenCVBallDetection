import cv2 as cv
import numpy as np
import time

# Ruta del video (modifica esto con tu archivo de video)
video_path = "video5.mp4"

# Cargar el video
captureVideo = cv.VideoCapture(video_path)

# Obtener tamaño original del video
original_width = int(captureVideo.get(cv.CAP_PROP_FRAME_WIDTH))
original_height = int(captureVideo.get(cv.CAP_PROP_FRAME_HEIGHT))

# Obtener FPS reales del video y calcular el tiempo entre frames
fps = captureVideo.get(cv.CAP_PROP_FPS)
frame_time = int(1000 / fps)  # Tiempo entre frames en milisegundos

# Conversión de píxeles a metros (ajusta esto según la escala de tu video)
pixel_to_meter = 1 / 100  # 1 píxel equivale a 1 cm (ajusta si es necesario)

# Filtro de Kalman optimizado
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

prev_positions = []
prev_time = None  
speed_measurements = []

lower_blue = np.array([100, 150, 50]) 
upper_blue = np.array([130, 255, 255])

lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

while True:
    ret, frame = captureVideo.read()
    if not ret:
        break 

    frame = cv.resize(frame, (original_width, original_height))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    mask_orange = cv.inRange(hsv, lower_orange, upper_orange)

    mask = cv.bitwise_or(mask_blue, mask_orange)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    current_center = None
    min_area_threshold = 200  

    if contours:
        valid_contours = [c for c in contours if cv.contourArea(c) > min_area_threshold]

        if valid_contours:
           
            max_contour = max(valid_contours, key=cv.contourArea)
            (x, y), radius = cv.minEnclosingCircle(max_contour)

            if 5 < radius < 80: 
                current_center = (int(x), int(y))

            
                measurement = np.array([[np.float32(x)], [np.float32(y)]])
                kalman.correct(measurement)

              
                cv.circle(frame, current_center, int(radius), (0, 255, 0), 2)
                cv.circle(frame, current_center, 2, (0, 0, 255), 3)

                prev_positions.append(current_center)
                if len(prev_positions) > 10:
                    prev_positions.pop(0)

    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])
    cv.circle(frame, (pred_x, pred_y), 5, (255, 0, 0), -1)

    # Cálculo de la velocidad con suavizado
    if len(prev_positions) >= 2:
        current_time = cv.getTickCount() / cv.getTickFrequency()
        if prev_time is not None:
            displacement = dist(prev_positions[-1], prev_positions[-2])
            time_sec = current_time - prev_time

            if time_sec > 0:
                speed_m_s = (displacement / time_sec) * pixel_to_meter

                # Suavizado de velocidad usando el promedio de las últimas 5 mediciones
                speed_measurements.append(speed_m_s)
                if len(speed_measurements) > 5:
                    speed_measurements.pop(0)
                avg_speed = np.mean(speed_measurements)

                cv.putText(frame, f"Velocidad: {avg_speed:.2f} m/s", (20, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        prev_time = current_time

    # Mostrar imágenes
    cv.imshow("Detección de Pelota", frame)
    cv.imshow("Máscara de Color", mask)

    key = cv.waitKey(frame_time) & 0xFF
    if key == ord('q'):
        break  # Salir con la tecla 'q'

captureVideo.release()
cv.destroyAllWindows()
