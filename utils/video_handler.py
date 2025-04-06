import os
import subprocess
import json
import base64
from typing import Dict, Any, List
from PIL import Image
import io
import tempfile

def extract_frames(video_path: str, interval_secs: int = 5) -> List[str]:
    """Extrae frames usando FFmpeg directamente"""
    frames = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Obtener duración del video
        duration = float(get_video_info(video_path).get('duration', 0))
        
        # Extraer frames cada interval_secs segundos
        for t in range(0, int(duration), interval_secs):
            output_frame = os.path.join(temp_dir, f"frame_{t}.jpg")
            
            cmd = [
                'ffmpeg', '-ss', str(t),
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '2',
                '-y', output_frame
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Convertir a base64
            if os.path.exists(output_frame):
                with Image.open(output_frame) as img:
                    img = img.resize((640, 360))
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    frames.append(img_str)
                os.remove(output_frame)
                
    except Exception as e:
        print(f"Error extracting frames: {e}")
    finally:
        # Limpiar archivos temporales
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
    return frames

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Obtiene información del video usando ffprobe"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in data.get('streams', []) 
             if s.get('codec_type') == 'video'),
            None
        )
        
        if not video_stream:
            return {'error': 'No video stream found'}
            
        return {
            'duration': float(data['format'].get('duration', 0)),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1'))
        }
        
    except Exception as e:
        print(f"Error getting video info: {e}")
        return {'error': str(e)}
