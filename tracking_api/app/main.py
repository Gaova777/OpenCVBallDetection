from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
from video_tracking import process_video

app = FastAPI()
VIDEO_FOLDER = "videos/"

@app.get("/process_video/")
async def process_local_video():
    video_filename = "video3.mp4"
    output_filename = "output.mp4"
    video_path = os.path.join(VIDEO_FOLDER, video_filename)
    output_path = os.path.join(VIDEO_FOLDER, output_filename)

    if not os.path.exists(video_path):
        return {"error": "El archivo de video no existe"}

    processed_video_path = process_video(video_path, output_path)
    
    return {"message": "Procesamiento completado", "download_url": f"/download_video/{output_filename}"}

@app.get("/download_video/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(VIDEO_FOLDER, filename)
    if not os.path.exists(file_path):
        return {"error": "El archivo no existe"}
    return FileResponse(file_path, media_type="video/mp4", filename=filename)
