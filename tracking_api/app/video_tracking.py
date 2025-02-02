import cv2 as cv
import numpy as np
import time
from kalman_filter import initialize_kalman
from ball_detection import detect_ball
from utils import calculate_speed, draw_tracking_info

def process_video(video_path):
    capture = cv.VideoCapture(video_path)
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_time = int(1000 / fps)

    kalman = initialize_kalman()

    lower_colors = [np.array([100, 150, 50]), np.array([10, 100, 100])]
    upper_colors = [np.array([130, 255, 255]), np.array([25, 255, 255])]

    prev_positions = []
    prev_time = None
    speed_measurements = []

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv.resize(frame, (width, height))
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        ball_position, mask = detect_ball(frame, hsv, lower_colors, upper_colors)

        if ball_position:
            measurement = np.array([[np.float32(ball_position[0])], [np.float32(ball_position[1])]])
            kalman.correct(measurement)
            prev_positions.append(ball_position)
            if len(prev_positions) > 10:
                prev_positions.pop(0)

        prediction = kalman.predict()

        speed, prev_time = calculate_speed(prev_positions, prev_time, 1/100)
        if speed is not None:
            speed_measurements.append(speed)
            if len(speed_measurements) > 5:
                speed_measurements.pop(0)
            avg_speed = np.mean(speed_measurements)
        else:
            avg_speed = None

        draw_tracking_info(frame, ball_position, prediction, avg_speed)

        cv.imshow("Detección de Pelota", frame)
        cv.imshow("Máscara de Color", mask)

        key = cv.waitKey(frame_time) & 0xFF
        if key == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
