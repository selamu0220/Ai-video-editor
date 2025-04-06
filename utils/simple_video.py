import os
import json
import base64
import subprocess
from PIL import Image
import io
from typing import List, Dict, Any

def extract_frames(video_path: str, interval_secs: int = 5) -> List[str]:
    """Extrae frames usando FFmpeg"""
    frames = []
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps=1/{interval_secs}',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-'
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        # Convertir el output en frames
        img_data = result.stdout
        while img_data:
            try:
                img = Image.open(io.BytesIO(img_data))
                img = img.resize((640, 360))
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                frames.append(base64.b64encode(buffer.getvalue()).decode())
                break  # Por ahora solo tomamos el primer frame
            except:
                break
                
        return frames
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Obtiene info básica del video usando FFprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {}
