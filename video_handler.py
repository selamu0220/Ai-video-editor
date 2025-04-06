import os
from typing import List, Dict, Any, Optional
from moviepy.editor import VideoFileClip
import numpy as np
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)

def extract_frames(video_path: str, num_frames: int = 5) -> List[str]:
    """Extrae frames usando moviepy"""
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            times = np.linspace(0, duration, num_frames)
            frames = []
            
            for t in times:
                # Obtener frame
                frame = clip.get_frame(t)
                
                # Convertir a PIL y redimensionar
                img = Image.fromarray(frame)
                img = img.resize((640, 360), Image.Resampling.LANCZOS)
                
                # Convertir a base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=80)
                b64_str = base64.b64encode(buffer.getvalue()).decode()
                frames.append(b64_str)
            
            return frames
    except Exception as e:
        logger.error(f"Error extrayendo frames: {e}")
        return []

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Obtiene información del video"""
    try:
        with VideoFileClip(video_path) as clip:
            return {
                'duration': clip.duration,
                'fps': clip.fps,
                'size': os.path.getsize(video_path),
                'width': int(clip.size[0]),
                'height': int(clip.size[1])
            }
    except Exception as e:
        logger.error(f"Error obteniendo info: {e}")
        return {}
