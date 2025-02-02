from fastapi import FastAPI
from video_tracking import process_video
import os

app = FastAPI()

# Carpeta donde se almacenan los videos en el servidor
VIDEO_FOLDER = "videos/"

# Asegurar que la carpeta de videos exista
os.makedirs(VIDEO_FOLDER, exist_ok=True)

@app.get("/process_video/")
async def process_local_video():
    """
    Procesa automáticamente un video almacenado en el servidor.
    """
    video_filename = "video5.mp4"  # Cambia esto si tienes otro video
    video_path = os.path.join(VIDEO_FOLDER, video_filename)

    if not os.path.exists(video_path):
        return {"error": f"El archivo {video_filename} no existe en {VIDEO_FOLDER}"}

    # Llamar a la función de procesamiento
    process_video(video_path)
    
    return {"message": f"Procesamiento de {video_filename} completado"}
