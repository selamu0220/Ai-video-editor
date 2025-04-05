"""
Componente de línea de tiempo interactiva para el editor de video AI
"""
import json
import streamlit as st
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import tempfile
import os
import subprocess
import base64
import io
from PIL import Image
import cv2

class InteractiveTimeline:
    def __init__(self, video_path: Optional[str] = None):
        """Inicializa la línea de tiempo interactiva"""
        self.segments = []
        self.markers = []
        self.effects = []
        self.transitions = []
        self.selected_segment = None
        self.selected_effect = None
        self.playhead_position = 0.0  # posición actual en segundos
        self.duration = 0
        self.fps = 0
        self.total_frames = 0
        self.video_path = video_path
        self.thumbnails = {}  # Cache de miniaturas para los segmentos
        
        # Si se proporciona una ruta de video, inicializar con ese video
        if video_path and os.path.exists(video_path):
            self.initialize_from_video(video_path)
    
    def initialize_from_video(self, video_path: str) -> None:
        """Inicializa la línea de tiempo a partir de un archivo de video usando SOLO ffprobe"""
        try:
            # Ejecutar ffprobe para obtener información del video
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            ffprobe_data = json.loads(result.stdout)
            
            # Extraer información relevante
            video_stream = None
            for stream in ffprobe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                 raise ValueError("No se encontró stream de video en el archivo.")

            # Calcular FPS
            fps = 0
            fps_str = video_stream.get('avg_frame_rate', video_stream.get('r_frame_rate', '0/1'))
            try:
                 parts = fps_str.split('/')
                 if len(parts) == 2 and int(parts[1]) != 0:
                      fps = float(int(parts[0]) / int(parts[1]))
            except Exception:
                 print(f"Warning: No se pudo parsear fps '{fps_str}'. Usando 0.")
                 fps = 0 # Default if parsing fails

            # Extraer dimensiones
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            
            # Duración en segundos (check format first, then stream)
            duration_str = ffprobe_data.get('format', {}).get('duration') or video_stream.get('duration')
            duration = float(duration_str) if duration_str else 0.0
            if duration <= 0:
                print(f"Warning: Duración inválida ({duration}) obtenida de ffprobe. Revisar video: {video_path}")
                # Maybe raise error or set a default? Let's set a small default for now
                duration = 1.0 # Avoid division by zero later

            # Establecer propiedades del timeline
            self.video_path = video_path
            self.duration = duration
            self.fps = fps if fps > 0 else 30 # Use a default fps if calculation failed
            self.total_frames = int(self.duration * self.fps)
            
            # Añadir el video completo como un segmento
            segment_id = 1 # Assuming initialization always starts with one segment
            
            self.segments = [{
                "id": segment_id,
                "start_time": 0,
                "end_time": self.duration,
                "type": "video",
                "source": video_path,
                "speed": 1.0,
                "label": "Clip principal",
                "color": "#4CAF50"
            }]
            
            # Generar miniatura para el segmento usando ffmpeg
            self.generate_thumbnail(segment_id, video_path, self.duration) # Pass necessary info

        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, KeyError, FileNotFoundError) as e:
             # Log detailed error
             import traceback
             print(f"Error inicializando timeline con ffprobe para {video_path}:")
             traceback.print_exc()
             # Re-raise a more user-friendly error or specific error for UI handling
             raise Exception(f"Error al procesar el video con ffprobe: {e}. Asegúrese que FFmpeg/ffprobe está instalado y accesible.")

    def generate_thumbnail(self, segment_id: int, video_path: str, duration: float) -> None:
        """Genera una miniatura para un segmento usando SOLO ffmpeg"""
        try:
            if not video_path or not os.path.exists(video_path) or duration <= 0:
                print(f"Warning: No se puede generar thumbnail para seg {segment_id}. Path: {video_path}, Duration: {duration}")
                return

            # Obtener tiempo del frame (20% del clip o 1s, lo que sea menor y válido)
            frame_time = min(max(0, duration * 0.2), 1.0) 
            # Ensure frame_time is not >= duration
            frame_time = min(frame_time, duration - 0.01 if duration > 0.01 else 0) 

            # Crear archivo temporal para el frame
            # Use context manager for safety if possible, otherwise ensure deletion
            temp_frame = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_frame_path = temp_frame.name
            temp_frame.close() # Close it so ffmpeg can write to it
            
            thumbnail_generated = False
            try:
                # Extraer el frame con ffmpeg
                extract_cmd = [
                    'ffmpeg',
                    '-ss', str(frame_time),
                    '-i', video_path,
                    '-vf', 'scale=160:-1', # Scale directly with ffmpeg
                    '-vframes', '1',
                    '-q:v', '3', # Quality setting (lower is better)
                    '-y', # Overwrite if exists
                    temp_frame_path
                ]
                
                print(f"Running thumbnail cmd: {' '.join(extract_cmd)}")
                subprocess.run(extract_cmd, capture_output=True, check=True, timeout=10) # Added timeout

                # Cargar y procesar la imagen SI el comando tuvo éxito
                if os.path.exists(temp_frame_path) and os.path.getsize(temp_frame_path) > 0:
                     img = cv2.imread(temp_frame_path)
                     if img is not None:
                          # Convertir de BGR a RGB (PIL expects RGB)
                          img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                          pil_img = Image.fromarray(img_rgb)
                          # No need to resize again if ffmpeg scaling worked
                          
                          # Guardar en el cache
                          buffered = io.BytesIO()
                          pil_img.save(buffered, format="JPEG")
                          img_str = base64.b64encode(buffered.getvalue()).decode()
                          self.thumbnails[segment_id] = img_str
                          thumbnail_generated = True
                     else:
                          print(f"Warning: ffmpeg created empty/invalid thumbnail file: {temp_frame_path}")
                else:
                     print(f"Warning: ffmpeg command ran but thumbnail file not found or empty: {temp_frame_path}")

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ffmpeg_err:
                 print(f"Error ejecutando ffmpeg para thumbnail seg {segment_id}: {ffmpeg_err}")
            except Exception as proc_err:
                 print(f"Error procesando thumbnail file para seg {segment_id}: {proc_err}")
            finally:
                 # Ensure temporary file is always deleted
                 if os.path.exists(temp_frame_path):
                      try:
                           os.unlink(temp_frame_path)
                      except OSError as e:
                           print(f"Warning: No se pudo borrar archivo temporal de thumbnail {temp_frame_path}: {e}")

            if not thumbnail_generated:
                print(f"Fallo al generar thumbnail para segmento {segment_id}. Usando placeholder.")
                # Optionally set a placeholder thumbnail if generation fails entirely
                # self.thumbnails[segment_id] = PLACEHOLDER_BASE64_STRING

        except Exception as e:
            import traceback
            print(f"Error inesperado generando thumbnail para seg {segment_id}:")
            traceback.print_exc()
    
    def add_segment(self, segment: Dict[str, Any]) -> None:
        """Añade un segmento a la línea de tiempo"""
        if "id" not in segment:
            segment_id = 1 if not self.segments else max([s.get("id", 0) for s in self.segments]) + 1
            segment["id"] = segment_id
        
        self.segments.append(segment)
        
        # Actualizar duración si es necesario
        segment_end = segment.get("end_time", 0)
        if segment_end > self.duration:
            self.duration = segment_end
        
        # Generar miniatura si es posible
        if "source" in segment and os.path.exists(segment["source"]):
            self.generate_thumbnail(segment["id"], segment["source"], segment.get("end_time", self.duration))
    
    def add_marker(self, time: float, label: str, color: str = "#FF5722") -> None:
        """Añade un marcador en un tiempo específico"""
        marker_id = 1 if not self.markers else max([m.get("id", 0) for m in self.markers]) + 1
        
        self.markers.append({
            "id": marker_id,
            "time": time,
            "label": label,
            "color": color
        })
    
    def add_effect(self, effect_type: str, start_time: float, end_time: float, 
                  params: Dict[str, Any], label: Optional[str] = None, color: str = "#9C27B0") -> None:
        """Añade un efecto a la línea de tiempo"""
        effect_id = 1 if not self.effects else max([e.get("id", 0) for e in self.effects]) + 1
        
        if label is None:
            label = effect_type
        
        self.effects.append({
            "id": effect_id,
            "type": effect_type,
            "start_time": start_time,
            "end_time": end_time,
            "params": params,
            "label": label,
            "color": color
        })
    
    def add_transition(self, transition_type: str, time: float, duration: float = 1.0,
                      params: Dict[str, Any] = {}, label: Optional[str] = None, color: str = "#2196F3") -> None:
        """Añade una transición a la línea de tiempo"""
        transition_id = 1 if not self.transitions else max([t.get("id", 0) for t in self.transitions]) + 1
        
        if label is None:
            label = transition_type
        
        self.transitions.append({
            "id": transition_id,
            "type": transition_type,
            "time": time,
            "duration": duration,
            "params": params,
            "label": label,
            "color": color
        })
    
    def split_segment_at(self, time: float) -> bool:
        """Divide un segmento en el tiempo especificado"""
        for segment in self.segments:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", self.duration)
            
            if start_time < time < end_time:
                # Crear dos nuevos segmentos
                segment1 = segment.copy()
                segment2 = segment.copy()
                
                # Actualizar los tiempos
                segment1["end_time"] = time
                segment2["start_time"] = time
                
                # Generar nuevo ID para el segundo segmento
                segment2["id"] = max([s.get("id", 0) for s in self.segments]) + 1
                
                # Actualizar la etiqueta
                segment1["label"] = f"{segment['label']} (parte 1)"
                segment2["label"] = f"{segment['label']} (parte 2)"
                
                # Eliminar el segmento original y añadir los nuevos
                self.segments.remove(segment)
                self.segments.append(segment1)
                self.segments.append(segment2)
                
                # Generar miniaturas si es posible
                if "source" in segment and os.path.exists(segment["source"]):
                    # Tomar fotogramas en los puntos medios de cada segmento
                    mid_point1 = (start_time + time) / 2
                    mid_point2 = (time + end_time) / 2
                    
                    # Generar las miniaturas
                    frame1 = clip.get_frame(mid_point1)
                    frame2 = clip.get_frame(mid_point2)
                    
                    # Convertir a base64
                    for idx, frame in [(segment1["id"], frame1), (segment2["id"], frame2)]:
                        pil_img = Image.fromarray(frame)
                        pil_img.thumbnail((160, 90))
                        buffered = io.BytesIO()
                        pil_img.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        self.thumbnails[idx] = img_str
                    
                    clip.close()
                
                return True
        
        return False
    
    def delete_segment(self, segment_id: int) -> bool:
        """Elimina un segmento de la línea de tiempo"""
        for segment in self.segments:
            if segment.get("id") == segment_id:
                self.segments.remove(segment)
                
                # Eliminar miniatura del cache
                if segment_id in self.thumbnails:
                    del self.thumbnails[segment_id]
                
                return True
        
        return False
    
    def delete_effect(self, effect_id: int) -> bool:
        """Elimina un efecto de la línea de tiempo"""
        for effect in self.effects:
            if effect.get("id") == effect_id:
                self.effects.remove(effect)
                return True
        
        return False
    
    def delete_marker(self, marker_id: int) -> bool:
        """Elimina un marcador de la línea de tiempo"""
        for marker in self.markers:
            if marker.get("id") == marker_id:
                self.markers.remove(marker)
                return True
        
        return False
    
    def delete_transition(self, transition_id: int) -> bool:
        """Elimina una transición de la línea de tiempo"""
        for transition in self.transitions:
            if transition.get("id") == transition_id:
                self.transitions.remove(transition)
                return True
        
        return False
    
    def update_segment(self, segment_id: int, changes: Dict[str, Any]) -> bool:
        """Actualiza un segmento con los cambios especificados"""
        for i, segment in enumerate(self.segments):
            if segment.get("id") == segment_id:
                # Actualizar el segmento con los cambios
                for key, value in changes.items():
                    segment[key] = value
                
                # Si cambió la fuente, regenerar la miniatura
                if "source" in changes and os.path.exists(changes["source"]):
                    self.generate_thumbnail(segment["id"], changes["source"], segment.get("end_time", self.duration))
                
                return True
        
        return False
    
    def update_effect(self, effect_id: int, changes: Dict[str, Any]) -> bool:
        """Actualiza un efecto con los cambios especificados"""
        for effect in self.effects:
            if effect.get("id") == effect_id:
                # Actualizar el efecto con los cambios
                for key, value in changes.items():
                    effect[key] = value
                return True
        
        return False
    
    def set_playhead_position(self, time: float) -> None:
        """Establece la posición del cabezal de reproducción"""
        self.playhead_position = max(0, min(time, self.duration))
    
    def get_segment_at_time(self, time: float) -> Optional[Dict[str, Any]]:
        """Obtiene el segmento en un tiempo específico"""
        for segment in self.segments:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", self.duration)
            
            if start_time <= time < end_time:
                return segment
        
        return None
    
    def get_effects_at_time(self, time: float) -> List[Dict[str, Any]]:
        """Obtiene los efectos activos en un tiempo específico"""
        active_effects = []
        
        for effect in self.effects:
            start_time = effect.get("start_time", 0)
            end_time = effect.get("end_time", self.duration)
            
            if start_time <= time < end_time:
                active_effects.append(effect)
        
        return active_effects
    
    def get_serializable_data(self) -> Dict[str, Any]:
        """Obtiene los datos de la línea de tiempo en un formato serializable"""
        return {
            "segments": self.segments,
            "markers": self.markers,
            "effects": self.effects,
            "transitions": self.transitions,
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "video_path": self.video_path
        }
    
    def from_serializable_data(self, data: Dict[str, Any]) -> None:
        """Carga la línea de tiempo desde datos serializables"""
        self.segments = data.get("segments", [])
        self.markers = data.get("markers", [])
        self.effects = data.get("effects", [])
        self.transitions = data.get("transitions", [])
        self.duration = data.get("duration", 0)
        self.fps = data.get("fps", 0)
        self.total_frames = data.get("total_frames", 0)
        self.video_path = data.get("video_path")
        
        # Regenerar miniaturas si se especifica la ruta del video
        if self.video_path and os.path.exists(self.video_path):
            clip = VideoFileClip(self.video_path)
            for segment in self.segments:
                self.generate_thumbnail(segment.get("id"), self.video_path, segment.get("end_time", self.duration))
            clip.close()

    def generate_thumbnails_for_all_segments(self) -> None:
        """Regenera miniaturas para todos los segmentos existentes"""
        if not self.video_path or not os.path.exists(self.video_path):
             print("No se puede generar miniaturas, falta video_path")
             return
             
        print("Regenerando todas las miniaturas...")
        for segment in self.segments:
             segment_id = segment.get("id")
             source_path = segment.get("source", self.video_path) # Use segment source if available, else main path
             duration = segment.get("end_time", self.duration) - segment.get("start_time", 0)
             if segment_id is not None and os.path.exists(source_path):
                  # The generate_thumbnail call already uses ffmpeg directly
                  self.generate_thumbnail(segment_id, source_path, duration) 
        print("Regeneración de miniaturas completada.")

def render_interactive_timeline(timeline: InteractiveTimeline, key_prefix: str = "timeline") -> Dict[str, Any]:
    """
    Renderiza la línea de tiempo interactiva en Streamlit
    
    Args:
        timeline: Objeto InteractiveTimeline para renderizar
        key_prefix: Prefijo para las claves de los widgets (útil si hay múltiples líneas de tiempo)
        
    Returns:
        Diccionario con información de interacción (eventos)
    """
    # Configuración de la línea de tiempo
    timeline_width = 800
    timeline_height = 200  # Aumentamos la altura para más espacio
    
    # Usar columnas para estructura
    col1, col2 = st.columns([7, 3])
    
    # Eventos que se devolverán
    events = {
        "selected_segment": None,
        "selected_effect": None,
        "playhead_moved": False,
        "segment_split": False,
        "split_time": None,
        "segment_deleted": None,
        "effect_deleted": None,
        "marker_deleted": None,
        "transition_deleted": None
    }
    
    # Calcular píxeles por segundo
    pixels_per_second = timeline_width / max(1, timeline.duration)
    
    # Utilizar un contenedor para la línea de tiempo
    with col1:
        timeline_container = st.container()
        st.markdown(f"<h4>Línea de Tiempo - Duración: {timeline.duration:.2f}s - FPS: {timeline.fps:.2f}</h4>", unsafe_allow_html=True)
        
        # Mostrar controles de navegación
        tcol1, tcol2, tcol3 = st.columns([1, 3, 1])
        with tcol1:
            if st.button("⏮ Inicio", key=f"{key_prefix}_goto_start"):
                timeline.set_playhead_position(0)
                events["playhead_moved"] = True
        
        with tcol2:
            # Slider para la posición del cabezal con comprobación de duración válida
            max_duration = max(0.1, float(timeline.duration))  # Asegurar que max_value sea mayor que min_value
            playhead_pos = st.slider(
                "Posición (segundos)",
                min_value=0.0,
                max_value=max_duration,
                value=min(float(timeline.playhead_position), max_duration),
                step=0.1,
                key=f"{key_prefix}_playhead"
            )
            
            if playhead_pos != timeline.playhead_position:
                timeline.set_playhead_position(playhead_pos)
                events["playhead_moved"] = True
        
        with tcol3:
            if st.button("⏭ Fin", key=f"{key_prefix}_goto_end"):
                timeline.set_playhead_position(timeline.duration)
                events["playhead_moved"] = True
        
        # Mostrar información del punto actual
        current_segment = timeline.get_segment_at_time(timeline.playhead_position)
        current_effects = timeline.get_effects_at_time(timeline.playhead_position)
        
        info_text = f"Posición: {timeline.playhead_position:.2f}s"
        if current_segment:
            segment_progress = (timeline.playhead_position - current_segment.get("start_time", 0)) / (
                current_segment.get("end_time", timeline.duration) - current_segment.get("start_time", 0)
            ) * 100
            info_text += f" | Clip: {current_segment.get('label', 'Sin nombre')} ({segment_progress:.1f}%)"
        
        if current_effects:
            effect_names = ", ".join([e.get("label", e.get("type", "Efecto")) for e in current_effects])
            info_text += f" | Efectos activos: {effect_names}"
        
        st.markdown(f"<div style='margin-bottom:10px;'>{info_text}</div>", unsafe_allow_html=True)
        
        # Crear HTML para la línea de tiempo visual
        timeline_html = f"""
        <div style="
            width: {timeline_width}px;
            height: {timeline_height}px;
            background-color: #2D2D2D;
            position: relative;
            border-radius: 5px;
            margin-bottom: 20px;
            overflow: hidden;
        ">
        """
        
        # Dibujar regla de tiempo
        for i in range(int(timeline.duration) + 1):
            if i % 5 == 0:  # Marcador principal cada 5 segundos
                height = 15
                width = 2
                color = "#FFFFFF"
                show_label = True
            else:  # Marcador secundario
                height = 8
                width = 1
                color = "#AAAAAA"
                show_label = False
            
            pos = i * pixels_per_second
            
            timeline_html += f"""
            <div style="
                position: absolute;
                left: {pos}px;
                top: 0;
                width: {width}px;
                height: {height}px;
                background-color: {color};
            "></div>
            """
            
            if show_label:
                timeline_html += f"""
                <div style="
                    position: absolute;
                    left: {pos - 10}px;
                    top: {height + 2}px;
                    width: 20px;
                    text-align: center;
                    font-size: 10px;
                    color: #FFFFFF;
                ">{i}s</div>
                """
        
        # Dibujar segmentos
        segment_height = 60
        segment_top = 30
        
        for segment in timeline.segments:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", timeline.duration)
            segment_id = segment.get("id", 0)
            segment_label = segment.get("label", "Clip")
            segment_color = segment.get("color", "#4CAF50")
            
            # Calcular posición y ancho
            left_pos = start_time * pixels_per_second
            width = (end_time - start_time) * pixels_per_second
            
            # Si es el segmento seleccionado, añadir borde
            border = "2px solid #FFFFFF" if segment_id == timeline.selected_segment else "none"
            
            # Obtener miniatura si existe
            thumbnail = ""
            if segment_id in timeline.thumbnails:
                img_data = timeline.thumbnails[segment_id]
                thumbnail = f"""
                <div style="
                    position: absolute;
                    left: 2px;
                    top: 2px;
                    width: calc(100% - 4px);
                    height: calc(100% - 20px);
                    background-image: url('data:image/jpeg;base64,{img_data}');
                    background-size: cover;
                    background-position: center;
                    opacity: 0.7;
                "></div>
                """
            
            timeline_html += f"""
            <div id="segment_{segment_id}" style="
                position: absolute;
                left: {left_pos}px;
                top: {segment_top}px;
                width: {width}px;
                height: {segment_height}px;
                background-color: {segment_color};
                border: {border};
                border-radius: 3px;
                overflow: hidden;
            ">
                {thumbnail}
                <div style="
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    background-color: rgba(0,0,0,0.7);
                    color: white;
                    font-size: 11px;
                    padding: 2px;
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{segment_label}</div>
            </div>
            """
        
        # Dibujar efectos
        effect_height = 25
        effect_top = segment_top + segment_height + 10
        
        for effect in timeline.effects:
            start_time = effect.get("start_time", 0)
            end_time = effect.get("end_time", timeline.duration)
            effect_id = effect.get("id", 0)
            effect_type = effect.get("type", "effect")
            effect_label = effect.get("label", effect_type)
            effect_color = effect.get("color", "#9C27B0")
            
            # Calcular posición y ancho
            left_pos = start_time * pixels_per_second
            width = (end_time - start_time) * pixels_per_second
            
            # Si es el efecto seleccionado, añadir borde
            border = "2px solid #FFFFFF" if effect_id == timeline.selected_effect else "none"
            
            timeline_html += f"""
            <div id="effect_{effect_id}" style="
                position: absolute;
                left: {left_pos}px;
                top: {effect_top}px;
                width: {width}px;
                height: {effect_height}px;
                background-color: {effect_color};
                border: {border};
                border-radius: 3px;
            ">
                <div style="
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    color: white;
                    font-size: 10px;
                    padding: 2px;
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{effect_label}</div>
            </div>
            """
        
        # Dibujar transiciones
        transition_height = 20
        transition_top = effect_top + effect_height + 10
        
        for transition in timeline.transitions:
            time = transition.get("time", 0)
            duration = transition.get("duration", 1.0)
            transition_id = transition.get("id", 0)
            transition_type = transition.get("type", "transition")
            transition_label = transition.get("label", transition_type)
            transition_color = transition.get("color", "#2196F3")
            
            # Calcular posición y ancho
            left_pos = (time - duration/2) * pixels_per_second
            width = duration * pixels_per_second
            
            timeline_html += f"""
            <div id="transition_{transition_id}" style="
                position: absolute;
                left: {left_pos}px;
                top: {transition_top}px;
                width: {width}px;
                height: {transition_height}px;
                background-color: {transition_color};
                border-radius: 3px;
            ">
                <div style="
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    color: white;
                    font-size: 9px;
                    padding: 1px;
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{transition_label}</div>
            </div>
            """
        
        # Dibujar marcadores
        for marker in timeline.markers:
            time = marker.get("time", 0)
            marker_id = marker.get("id", 0)
            label = marker.get("label", "Marker")
            color = marker.get("color", "#FF5722")
            
            # Calcular posición
            left_pos = time * pixels_per_second
            
            timeline_html += f"""
            <div id="marker_{marker_id}" style="
                position: absolute;
                left: {left_pos}px;
                top: 0;
                width: 2px;
                height: {timeline_height}px;
                background-color: {color};
            "></div>
            <div style="
                position: absolute;
                left: {left_pos - 50}px;
                top: {timeline_height - 15}px;
                width: 100px;
                text-align: center;
                font-size: 10px;
                color: {color};
                transform: rotate(-45deg);
                transform-origin: 50% 0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            ">{label}</div>
            """
        
        # Dibujar el cabezal de reproducción
        playhead_pos = timeline.playhead_position * pixels_per_second
        
        timeline_html += f"""
        <div id="playhead" style="
            position: absolute;
            left: {playhead_pos}px;
            top: 0;
            width: 3px;
            height: {timeline_height}px;
            background-color: #FFC107;
            z-index: 100;
        "></div>
        <div style="
            position: absolute;
            left: {playhead_pos - 8}px;
            top: 0;
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-top: 8px solid #FFC107;
            z-index: 101;
        "></div>
        """
        
        # Cerrar el div de la línea de tiempo
        timeline_html += "</div>"
        
        # Mostrar la línea de tiempo
        timeline_container.markdown(timeline_html, unsafe_allow_html=True)
        
        # Botones de acción para la línea de tiempo
        act_col1, act_col2, act_col3, act_col4 = st.columns(4)
        
        with act_col1:
            if st.button("✂️ Dividir en punto actual", key=f"{key_prefix}_split"):
                if timeline.split_segment_at(timeline.playhead_position):
                    events["segment_split"] = True
                    events["split_time"] = timeline.playhead_position
                    st.success(f"Segmento dividido en {timeline.playhead_position:.2f}s")
                else:
                    st.warning("No se pudo dividir ningún segmento en este punto")
        
        with act_col2:
            if st.button("🎨 Añadir marcador", key=f"{key_prefix}_add_marker"):
                marker_label = f"Marcador en {timeline.playhead_position:.2f}s"
                timeline.add_marker(timeline.playhead_position, marker_label)
                st.success(f"Marcador añadido en {timeline.playhead_position:.2f}s")
        
        with act_col3:
            if current_segment:
                segment_id = current_segment.get("id")
                if st.button(f"🗑️ Eliminar segmento", key=f"{key_prefix}_del_segment"):
                    if len(timeline.segments) > 1:
                        if timeline.delete_segment(segment_id):
                            events["segment_deleted"] = segment_id
                            st.success(f"Segmento eliminado")
                        else:
                            st.error(f"No se pudo eliminar el segmento")
                    else:
                        st.warning("No se puede eliminar el único segmento")
        
        with act_col4:
            if current_effects:
                effect_id = current_effects[0].get("id")
                if st.button(f"🗑️ Eliminar efecto", key=f"{key_prefix}_del_effect"):
                    if timeline.delete_effect(effect_id):
                        events["effect_deleted"] = effect_id
                        st.success(f"Efecto eliminado")
                    else:
                        st.error(f"No se pudo eliminar el efecto")
    
    # Panel lateral para detalles y edición
    with col2:
        st.markdown("### Detalles")
        
        # Detalles del segmento actual
        if current_segment:
            st.markdown(f"**Segmento:** {current_segment.get('label', 'Sin nombre')}")
            st.progress((timeline.playhead_position - current_segment.get("start_time", 0)) / 
                       (current_segment.get("end_time", timeline.duration) - current_segment.get("start_time", 0)))
            
            segment_info = f"""
            - **Inicio:** {current_segment.get('start_time', 0):.2f}s
            - **Fin:** {current_segment.get('end_time', timeline.duration):.2f}s
            - **Duración:** {current_segment.get('end_time', timeline.duration) - current_segment.get('start_time', 0):.2f}s
            - **Velocidad:** {current_segment.get('speed', 1.0)}x
            """
            st.markdown(segment_info)
            
            # Mostrar miniatura si existe
            segment_id = current_segment.get("id")
            if segment_id in timeline.thumbnails:
                st.markdown("**Vista previa:**")
                st.image(f"data:image/jpeg;base64,{timeline.thumbnails[segment_id]}")
        
        # Detalles de los efectos activos
        if current_effects:
            st.markdown("**Efectos activos:**")
            for effect in current_effects:
                effect_type = effect.get("type", "Efecto")
                effect_label = effect.get("label", effect_type)
                st.markdown(f"- {effect_label}")
                
                # Mostrar parámetros del efecto
                params = effect.get("params", {})
                if params:
                    param_text = "\n".join([f"  - {k}: {v}" for k, v in params.items()])
                    st.markdown(f"  Parámetros:\n{param_text}")
    
    return events

def initialize_timeline_from_video(video_path: str) -> InteractiveTimeline:
    """
    Inicializa una línea de tiempo a partir de un archivo de video
    
    Args:
        video_path: Ruta al archivo de video
        
    Returns:
        Objeto InteractiveTimeline inicializado
    """
    timeline = InteractiveTimeline()
    timeline.initialize_from_video(video_path)
    return timeline

def add_tool_panel(timeline: InteractiveTimeline) -> Dict[str, Any]:
    """
    Añade un panel de herramientas para trabajar con la línea de tiempo
    
    Args:
        timeline: Objeto InteractiveTimeline
        
    Returns:
        Diccionario con información de la herramienta seleccionada y parámetros
    """
    st.markdown("## 🧰 Herramientas de Edición")
    
    # Crear pestañas para las diferentes categorías de herramientas
    tabs = st.tabs([
        "✂️ Corte y División",
        "🎨 Efectos Visuales",
        "🔊 Audio",
        "🔄 Transiciones",
        "📝 Texto"
    ])
    
    tool_result = {
        "tool_selected": None,
        "parameters": {}
    }
    
    # Pestaña de Corte y División
    with tabs[0]:
        st.markdown("### Herramientas de Corte")
        
        operation = st.selectbox(
            "Operación",
            ["Seleccionar operación", "Recortar video", "Dividir en segmentos", "Eliminar segmento", "Modificar velocidad"]
        )
        
        if operation == "Recortar video":
            st.markdown("Recortar el video entre dos puntos específicos")
            
            # Obtener puntos de inicio y fin
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.number_input("Tiempo de inicio (s)", 
                                           min_value=0.0, 
                                           max_value=timeline.duration,
                                           value=timeline.playhead_position)
            
            with col2:
                end_time = st.number_input("Tiempo de fin (s)", 
                                         min_value=start_time, 
                                         max_value=timeline.duration,
                                         value=min(timeline.playhead_position + 5.0, timeline.duration))
            
            if st.button("Aplicar recorte"):
                tool_result["tool_selected"] = "trim_clip"
                tool_result["parameters"] = {
                    "start_time": start_time,
                    "end_time": end_time
                }
        
        elif operation == "Dividir en segmentos":
            st.markdown("Dividir el video en múltiples segmentos")
            
            # Obtener puntos de división
            split_points_str = st.text_input(
                "Puntos de división (segundos, separados por comas)",
                value=f"{timeline.playhead_position:.1f}"
            )
            
            try:
                split_points = [float(p.strip()) for p in split_points_str.split(",")]
                valid_points = [p for p in split_points if 0 < p < timeline.duration]
                
                if valid_points:
                    st.success(f"Se creará{'' if len(valid_points)==1 else 'n'} {len(valid_points)} punto{'' if len(valid_points)==1 else 's'} de división")
                    
                    if st.button("Aplicar división"):
                        tool_result["tool_selected"] = "split_clip"
                        tool_result["parameters"] = {
                            "split_points": valid_points
                        }
                else:
                    st.warning("No hay puntos de división válidos")
                
            except ValueError:
                st.error("Formato inválido. Ingresa números separados por comas.")
        
        elif operation == "Modificar velocidad":
            st.markdown("Cambiar la velocidad de reproducción del video")
            
            speed_factor = st.slider(
                "Factor de velocidad",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1
            )
            
            speed_preview = ""
            if speed_factor > 1.0:
                speed_preview = f"{speed_factor}x más rápido"
            elif speed_factor < 1.0:
                speed_preview = f"{1/speed_factor:.1f}x más lento"
            else:
                speed_preview = "Velocidad normal"
            
            st.info(speed_preview)
            
            if st.button("Aplicar cambio de velocidad"):
                tool_result["tool_selected"] = "change_speed"
                tool_result["parameters"] = {
                    "speed_factor": speed_factor
                }
    
    # Pestaña de Efectos Visuales
    with tabs[1]:
        st.markdown("### Efectos Visuales")
        
        effect_type = st.selectbox(
            "Tipo de efecto",
            ["Seleccionar efecto", "Filtro de color", "Blur (desenfoque)", "Rotación", "Espejo", "Ajuste de color"]
        )
        
        if effect_type == "Filtro de color":
            filter_type = st.selectbox(
                "Filtro",
                ["Escala de grises", "Sepia", "Alto contraste", "Vintage"]
            )
            
            filter_map = {
                "Escala de grises": "grayscale",
                "Sepia": "sepia", 
                "Alto contraste": "contrast",
                "Vintage": "vintage"
            }
            
            intensity = st.slider("Intensidad", 0.1, 2.0, 1.0, 0.1)
            
            if st.button("Aplicar filtro"):
                tool_result["tool_selected"] = "apply_color_filter"
                tool_result["parameters"] = {
                    "filter_type": filter_map[filter_type],
                    "intensity": intensity
                }
                
                # Añadir el efecto a la línea de tiempo para visualización
                timeline.add_effect(
                    effect_type=filter_map[filter_type],
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={"intensity": intensity},
                    label=f"Filtro: {filter_type}"
                )
        
        elif effect_type == "Blur (desenfoque)":
            radius = st.slider("Radio de desenfoque", 1, 20, 5)
            
            if st.button("Aplicar desenfoque"):
                tool_result["tool_selected"] = "apply_visual_effect"
                tool_result["parameters"] = {
                    "effect_type": "blur",
                    "params": {"radius": radius}
                }
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="blur",
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={"radius": radius},
                    label=f"Desenfoque: {radius}"
                )
        
        elif effect_type == "Rotación":
            angle = st.selectbox("Ángulo de rotación", [90, 180, 270])
            
            if st.button("Aplicar rotación"):
                tool_result["tool_selected"] = "apply_visual_effect"
                tool_result["parameters"] = {
                    "effect_type": "rotate",
                    "params": {"angle": angle}
                }
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="rotate",
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={"angle": angle},
                    label=f"Rotación: {angle}°"
                )
        
        elif effect_type == "Espejo":
            axis = st.radio("Eje", ["Horizontal", "Vertical"])
            axis_param = "x" if axis == "Horizontal" else "y"
            
            if st.button("Aplicar efecto espejo"):
                tool_result["tool_selected"] = "apply_visual_effect"
                tool_result["parameters"] = {
                    "effect_type": "mirror",
                    "params": {"axis": axis_param}
                }
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="mirror",
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={"axis": axis_param},
                    label=f"Espejo: {axis}"
                )
        
        elif effect_type == "Ajuste de color":
            col1, col2 = st.columns(2)
            
            with col1:
                brightness = st.slider("Brillo", 0.0, 2.0, 1.0, 0.1)
                contrast = st.slider("Contraste", 0.0, 2.0, 1.0, 0.1)
            
            with col2:
                saturation = st.slider("Saturación", 0.0, 2.0, 1.0, 0.1)
            
            if st.button("Aplicar ajuste de color"):
                tool_result["tool_selected"] = "color_adjustment"
                tool_result["parameters"] = {
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation
                }
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="color_adjustment",
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={
                        "brightness": brightness,
                        "contrast": contrast,
                        "saturation": saturation
                    },
                    label="Ajuste de color"
                )
    
    # Pestaña de Audio
    with tabs[2]:
        st.markdown("### Herramientas de Audio")
        
        audio_tool = st.selectbox(
            "Operación de audio",
            ["Seleccionar operación", "Ajustar volumen", "Eliminar audio", "Cortar silencios", "Añadir música"]
        )
        
        if audio_tool == "Ajustar volumen":
            volume_factor = st.slider("Factor de volumen", 0.0, 5.0, 1.0, 0.1)
            
            volume_preview = ""
            if volume_factor > 1.0:
                volume_preview = f"{volume_factor}x más alto"
            elif volume_factor < 1.0:
                volume_preview = f"{1/volume_factor:.1f}x más bajo"
            else:
                volume_preview = "Volumen normal"
            
            st.info(volume_preview)
            
            if st.button("Aplicar ajuste de volumen"):
                tool_result["tool_selected"] = "adjust_volume"
                tool_result["parameters"] = {
                    "volume_factor": volume_factor
                }
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="volume",
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={"factor": volume_factor},
                    label=f"Volumen: {volume_factor}x"
                )
        
        elif audio_tool == "Eliminar audio":
            if st.button("Eliminar todo el audio"):
                tool_result["tool_selected"] = "remove_audio"
                tool_result["parameters"] = {}
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="no_audio",
                    start_time=timeline.playhead_position,
                    end_time=timeline.duration,
                    params={},
                    label="Sin audio"
                )
        
        elif audio_tool == "Cortar silencios":
            col1, col2 = st.columns(2)
            
            with col1:
                min_silence_len = st.number_input("Duración mínima del silencio (ms)", 
                                                min_value=100, 
                                                max_value=5000,
                                                value=500,
                                                step=100)
            
            with col2:
                silence_thresh = st.slider("Umbral de silencio (dB)", 
                                         min_value=-60, 
                                         max_value=-20,
                                         value=-40,
                                         step=5)
            
            if st.button("Cortar silencios"):
                tool_result["tool_selected"] = "cut_silences"
                tool_result["parameters"] = {
                    "min_silence_len": min_silence_len,
                    "silence_thresh": silence_thresh
                }
                
                # Añadir el efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="cut_silences",
                    start_time=0,
                    end_time=timeline.duration,
                    params={
                        "min_silence_len": min_silence_len,
                        "silence_thresh": silence_thresh
                    },
                    label="Cortar silencios"
                )
    
    # Pestaña de Transiciones
    with tabs[3]:
        st.markdown("### Transiciones")
        
        transition_type = st.selectbox(
            "Tipo de transición",
            ["Seleccionar transición", "Fundido", "Fundido cruzado", "Deslizamiento", "Zoom"]
        )
        
        if transition_type == "Fundido":
            fade_type = st.radio("Tipo de fundido", ["Entrada", "Salida", "Ambos"])
            duration = st.slider("Duración (segundos)", 0.5, 5.0, 1.0, 0.5)
            
            fade_in = duration if fade_type in ["Entrada", "Ambos"] else 0
            fade_out = duration if fade_type in ["Salida", "Ambos"] else 0
            
            if st.button("Aplicar fundido"):
                tool_result["tool_selected"] = "add_fade_transition"
                tool_result["parameters"] = {
                    "fade_in": fade_in,
                    "fade_out": fade_out
                }
                
                # Añadir la transición a la línea de tiempo
                if fade_in > 0:
                    timeline.add_transition(
                        transition_type="fade_in",
                        time=0,
                        duration=fade_in,
                        label="Fundido entrada"
                    )
                
                if fade_out > 0:
                    timeline.add_transition(
                        transition_type="fade_out",
                        time=timeline.duration,
                        duration=fade_out,
                        label="Fundido salida"
                    )
        
        elif transition_type == "Fundido cruzado":
            st.markdown("Nota: Esta transición requiere múltiples clips")
            
            fade_duration = st.slider("Duración (segundos)", 0.5, 3.0, 1.0, 0.5)
            
            if st.button("Añadir fundido cruzado"):
                if len(timeline.segments) > 1:
                    tool_result["tool_selected"] = "add_crossfade_transition"
                    tool_result["parameters"] = {
                        "crossfade_duration": fade_duration
                    }
                    
                    # Añadir transiciones entre segmentos
                    for i in range(len(timeline.segments)-1):
                        segment = timeline.segments[i]
                        end_time = segment.get("end_time")
                        timeline.add_transition(
                            transition_type="crossfade",
                            time=end_time,
                            duration=fade_duration,
                            label="Fundido cruzado"
                        )
                else:
                    st.warning("Se necesitan al menos dos segmentos para usar fundido cruzado")
    
    # Pestaña de Texto
    with tabs[4]:
        st.markdown("### Texto y Subtítulos")
        
        text_tool = st.selectbox(
            "Herramienta de texto",
            ["Seleccionar herramienta", "Añadir texto", "Título animado", "Generar subtítulos"]
        )
        
        if text_tool == "Añadir texto":
            text = st.text_input("Texto a añadir", "Texto de ejemplo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                position_h = st.selectbox("Posición horizontal", ["Izquierda", "Centro", "Derecha"])
                fontsize = st.slider("Tamaño de fuente", 10, 100, 30, 5)
            
            with col2:
                position_v = st.selectbox("Posición vertical", ["Arriba", "Centro", "Abajo"])
                color = st.color_picker("Color del texto", "#FFFFFF")
            
            # Mapear a valores aceptados por la función
            position_map_h = {"Izquierda": "left", "Centro": "center", "Derecha": "right"}
            position_map_v = {"Arriba": "top", "Centro": "center", "Abajo": "bottom"}
            
            position = (position_map_h[position_h], position_map_v[position_v])
            
            # Duración del texto
            st.markdown("Duración del texto:")
            col1, col2 = st.columns(2)
            
            with col1:
                start_time = st.number_input("Tiempo de inicio (s)", 
                                           min_value=0.0,
                                           max_value=timeline.duration,
                                           value=timeline.playhead_position)
            
            with col2:
                duration = st.number_input("Duración (s)",
                                          min_value=1.0,
                                          max_value=timeline.duration - start_time,
                                          value=min(5.0, timeline.duration - start_time))
            
            if st.button("Añadir texto"):
                tool_result["tool_selected"] = "add_text"
                tool_result["parameters"] = {
                    "text": text,
                    "position": position,
                    "fontsize": fontsize,
                    "color": color,
                    "start_time": start_time,
                    "duration": duration
                }
                
                # Añadir efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="text",
                    start_time=start_time,
                    end_time=start_time + duration,
                    params={
                        "text": text,
                        "position": position,
                        "fontsize": fontsize,
                        "color": color
                    },
                    label=f"Texto: {text[:15]+'...' if len(text)>15 else text}"
                )
        
        elif text_tool == "Título animado":
            title = st.text_input("Texto del título", "Mi Vídeo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                animation = st.selectbox("Tipo de animación", ["Fade", "Slide", "Zoom"])
                fontsize = st.slider("Tamaño de fuente", 20, 150, 50, 5)
            
            with col2:
                duration = st.slider("Duración (segundos)", 1.0, 10.0, 5.0, 0.5)
                color = st.color_picker("Color del título", "#FFFFFF")
            
            animation_map = {"Fade": "fade", "Slide": "slide", "Zoom": "zoom"}
            
            if st.button("Añadir título animado"):
                tool_result["tool_selected"] = "add_animated_title"
                tool_result["parameters"] = {
                    "title": title,
                    "animation_type": animation_map[animation],
                    "duration": duration,
                    "fontsize": fontsize,
                    "color": color
                }
                
                # Añadir efecto a la línea de tiempo
                timeline.add_effect(
                    effect_type="animated_title",
                    start_time=0,
                    end_time=duration,
                    params={
                        "title": title,
                        "animation_type": animation_map[animation],
                        "fontsize": fontsize,
                        "color": color
                    },
                    label=f"Título: {title[:15]+'...' if len(title)>15 else title}"
                )
    
    return tool_result

def update_timeline_with_tool(timeline: InteractiveTimeline, tool_result: Dict[str, Any], video_path: str) -> Tuple[Optional[str], str]:
    """
    Actualiza la línea de tiempo aplicando la herramienta seleccionada
    
    Args:
        timeline: Objeto InteractiveTimeline
        tool_result: Diccionario con la herramienta y parámetros
        video_path: Ruta al video actual
        
    Returns:
        Tupla con (nueva ruta de video, mensaje de resultado)
    """
    from utils.video_editing_tools import (
        trim_clip, split_clip, change_speed, apply_color_filter, apply_visual_effect,
        add_text, add_animated_title, add_fade_transition, add_crossfade_transition,
        adjust_volume, remove_audio, cut_silences
    )
    
    tool = tool_result["tool_selected"]
    params = tool_result["parameters"]
    
    if not tool:
        return None, "No se seleccionó ninguna herramienta"
    
    temp_dir = tempfile.mkdtemp()
    result_message = "Operación completada"
    new_video_path = None
    
    try:
        if tool == "trim_clip":
            start_time = params["start_time"]
            end_time = params["end_time"]
            new_video_path = trim_clip(video_path, start_time, end_time)
            result_message = f"Video recortado de {start_time:.2f}s a {end_time:.2f}s"
        
        elif tool == "split_clip":
            split_points = params["split_points"]
            new_paths = split_clip(video_path, split_points, temp_dir)
            new_video_path = new_paths[0]  # Usar el primer clip como resultado
            result_message = f"Video dividido en {len(new_paths)} segmentos"
        
        elif tool == "change_speed":
            speed_factor = params["speed_factor"]
            new_video_path = change_speed(video_path, speed_factor)
            result_message = f"Velocidad de video cambiada a {speed_factor}x"
        
        elif tool == "apply_color_filter":
            filter_type = params["filter_type"]
            intensity = params["intensity"]
            new_video_path = apply_color_filter(video_path, filter_type, intensity)
            result_message = f"Filtro de color '{filter_type}' aplicado con intensidad {intensity}"
        
        elif tool == "apply_visual_effect":
            effect_type = params["effect_type"]
            effect_params = params["params"]
            new_video_path = apply_visual_effect(video_path, effect_type, effect_params)
            result_message = f"Efecto visual '{effect_type}' aplicado"
        
        elif tool == "color_adjustment":
            brightness = params["brightness"]
            contrast = params["contrast"]
            saturation = params["saturation"]
            
            from utils.video_processor import apply_color_adjustment
            clip = VideoFileClip(video_path)
            adjusted_clip = apply_color_adjustment(clip, brightness, contrast, saturation)
            
            new_video_path = os.path.join(temp_dir, f"color_adjusted.mp4")
            adjusted_clip.write_videofile(new_video_path, codec="libx264", audio_codec="aac")
            
            clip.close()
            adjusted_clip.close()
            
            result_message = f"Ajuste de color aplicado (brillo: {brightness}, contraste: {contrast}, saturación: {saturation})"
        
        elif tool == "add_text":
            text = params["text"]
            position = params["position"]
            fontsize = params["fontsize"]
            color = params["color"]
            start_time = params["start_time"]
            duration = params["duration"]
            
            new_video_path = add_text(
                video_path, text, position, fontsize, color, 
                duration, start_time
            )
            
            result_message = f"Texto '{text}' añadido al video"
        
        elif tool == "add_animated_title":
            title = params["title"]
            animation_type = params["animation_type"]
            duration = params["duration"]
            fontsize = params["fontsize"]
            color = params["color"]
            
            new_video_path = add_animated_title(
                video_path, title, animation_type, duration, fontsize, color
            )
            
            result_message = f"Título animado '{title}' añadido al video"
        
        elif tool == "add_fade_transition":
            fade_in = params["fade_in"]
            fade_out = params["fade_out"]
            
            new_video_path = add_fade_transition(video_path, fade_in, fade_out)
            
            fade_types = []
            if fade_in > 0:
                fade_types.append(f"entrada ({fade_in}s)")
            if fade_out > 0:
                fade_types.append(f"salida ({fade_out}s)")
            
            result_message = f"Fundido de {' y '.join(fade_types)} aplicado"
        
        elif tool == "add_crossfade_transition":
            # Esta operación requiere múltiples clips
            result_message = "Para aplicar fundido cruzado, primero divide el video en clips separados"
            new_video_path = None
        
        elif tool == "adjust_volume":
            volume_factor = params["volume_factor"]
            
            new_video_path = adjust_volume(video_path, volume_factor)
            
            result_message = f"Volumen ajustado a {volume_factor}x"
        
        elif tool == "remove_audio":
            new_video_path = remove_audio(video_path)
            result_message = "Audio eliminado del video"
        
        elif tool == "cut_silences":
            min_silence_len = params["min_silence_len"]
            silence_thresh = params["silence_thresh"]
            
            new_video_path = cut_silences(video_path, min_silence_len, silence_thresh)
            
            result_message = f"Silencios cortados (umbral: {silence_thresh}dB, mín. duración: {min_silence_len}ms)"
    
    except Exception as e:
        result_message = f"Error al aplicar la herramienta: {str(e)}"
        new_video_path = None
    
    return new_video_path, result_message