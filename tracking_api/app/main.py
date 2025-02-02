from fastapi import FastAPI
from video_tracking import process_video
import os

app = FastAPI()
VIDEO_FOLDER = "videos/"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

@app.get("/process_video/")
async def process_local_video():
   
    video_filename = "Video5.mp4"  
    video_path = os.path.join(VIDEO_FOLDER, video_filename)

    if not os.path.exists(video_path):
        return {"error": f"El archivo {video_filename} no existe en {VIDEO_FOLDER}"}
    process_video(video_path)
    
    return {"message": f"Procesamiento de {video_filename} completado"}
