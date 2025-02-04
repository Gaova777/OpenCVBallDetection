import cv2 as cv
import numpy as np

def initialize_kalman():
   #inicializa el filtro con los parametros de entrada neesarios para llamar los metodos de predicción. El filtro se inicializa con una matriz de 6 estados y 2 mediciones.
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
