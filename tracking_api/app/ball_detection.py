import cv2 as cv
import numpy as np

def detect_ball(frame, hsv, lower_colors, upper_colors):
    mask = None

    for lower, upper in zip(lower_colors, upper_colors):
        color_mask = cv.inRange(hsv, lower, upper)#Mascara binaria
        mask = color_mask if mask is None else cv.bitwise_or(mask, color_mask) # operacion OR para combinar las mascaras, obteniendo la mascara final. solo se mostrara en este caso la de azul.

#Mejora de la imagen con operaciones morfologicas ruido y erosiones
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        valid_contours = [c for c in contours if cv.contourArea(c) > 200] #filtro para contronos mayores de area de 200
        if valid_contours:
            max_contour = max(valid_contours, key=cv.contourArea)
            (x, y), radius = cv.minEnclosingCircle(max_contour) #calculo del circulo envolvente

            if 5 < radius < 80:
                return (int(x), int(y)), mask

    return None, mask
