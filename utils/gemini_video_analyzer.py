import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import base64
import json
from .video_handler import extract_frames, get_video_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_opencv():
    """Inicializa OpenCV con manejo robusto de errores"""
    try:
        # Intentar instalar dependencias del sistema primero
        subprocess.check_call(["apt-get", "update"], stderr=subprocess.DEVNULL)
        subprocess.check_call([
            "apt-get", "install", "-y",
            "python3-opencv",
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender1"
        ], stderr=subprocess.DEVNULL)
        
        # Desinstalar versiones existentes de OpenCV
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y",
            "opencv-python", "opencv-python-headless"
        ], stderr=subprocess.DEVNULL)
        
        # Instalar versión específica
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-cache-dir",
            "opencv-python-headless==4.5.5.64"
        ])
        
        import cv2
        logger.info(f"OpenCV {cv2.__version__} inicializado correctamente")
        return cv2
    except Exception as e:
        logger.error(f"Error crítico inicializando OpenCV: {e}")
        raise RuntimeError(f"No se pudo inicializar OpenCV: {e}")

# Inicializar OpenCV
cv2 = initialize_opencv()

import tempfile
import subprocess
import traceback
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import ffmpeg
import numpy as np
import shlex

# Funciones para extraer fotogramas y audio para enviar a Gemini

def extract_audio_sample(video_path: str, output_dir: str, duration: int = 15) -> Optional[str]:
    """
    Extrae una muestra de audio del video para analizar
    
    Args:
        video_path: Ruta al archivo de video
        output_dir: Directorio para guardar el audio
        duration: Duración máxima de la muestra en segundos
        
    Returns:
        Ruta al archivo de audio o None si falla
    """
    try:
        # Crear nombre para archivo temporal
        audio_path = os.path.join(output_dir, f"audio_sample_{os.path.basename(video_path)}.mp3")
        
        # Extraer audio con FFmpeg
        cmd = [
            "ffmpeg", "-i", video_path, 
            "-t", str(duration),  # Limitar duración
            "-q:a", "0",  # Alta calidad
            "-map", "a",  # Solo audio
            "-y",  # Sobrescribir
            audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path
        return None
    
    except Exception as e:
        print(f"Error extrayendo audio del video: {e}")
        traceback.print_exc()
        return None

def analyze_video_with_gemini(
    video_path: str, 
    user_prompt: str,
    num_frames: int = 5,
    temp_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Analiza video usando Gemini"""
    try:
        # Extraer frames usando el nuevo handler
        frames = extract_frames(video_path, num_frames)
        if not frames:
            return {"error": "No se pudieron extraer frames"}
            
        # Obtener info del video
        video_info = get_video_info(video_path)
        
        # Usar API key proporcionada o de variable de entorno
        if api_key:
            genai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        else:
            # HARDCODED API KEY
            api_key = "AIzaSyAPSwI9DbhxOR4OBMH7TMYB0ZYx3veUigY"
            genai.configure(api_key=api_key)
            print("Utilizando API key hardcoded para Gemini")
        
        # Crear directorio temporal si no se proporciona
        if not temp_dir:
            temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Extraer frames del video
            frames = extract_frames_from_video(video_path, num_frames)
            if not frames:
                return {"error": "No se pudieron extraer frames del video"}
            
            # 2. Obtener información técnica del video
            video_info = get_video_info_for_gemini(video_path)
            
            # 3. Preparar prompt para Gemini con las imágenes
            system_prompt = """
            Eres un asistente especializado en procesamiento y edición de video utilizando FFmpeg.
            
            Te voy a proporcionar:
            1. Varios fotogramas de un video
            2. Información técnica del video
            3. Una solicitud de edición o análisis
            
            INSTRUCCIONES:
            - Analiza los fotogramas para entender el contenido del video
            - Observa las características técnicas del video
            - Interpreta la solicitud del usuario y genera la respuesta apropiada
            - Si la solicitud implica editar el video, DEBES proporcionar un comando FFmpeg completo y detallado
            
            TU RESPUESTA DEBE TENER ESTE FORMATO (en JSON):
            {
                "análisis": "Descripción breve de lo que observas en el video",
                "evaluación_técnica": "Evaluación de la calidad técnica del video",
                "recomendación": "Tu recomendación basada en la solicitud",
                "comandos_ffmpeg": [
                    {
                        "descripción": "Explicación de lo que hace este comando",
                        "comando": "Comando FFmpeg completo",
                        "parámetros": {
                            // Explicación de los parámetros principales utilizados
                        }
                    }
                ]
            }
            
            Si el usuario pide un efecto específico que FFmpeg puede realizar, SIEMPRE incluye el comando correspondiente.
            """
            
            human_prompt = f"""
            FOTOGRAMAS DEL VIDEO: Te estoy mostrando {len(frames)} fotogramas representativos del video.
            
            INFORMACIÓN TÉCNICA DEL VIDEO:
            {json.dumps(video_info, indent=2)}
            
            SOLICITUD DEL USUARIO:
            "{user_prompt}"
            
            Responde con el JSON solicitado, incluyendo análisis y comandos FFmpeg necesarios.
            """
            
            # 4. Configurar modelo y configuración
            model = genai.GenerativeModel(
                "gemini-1.5-pro",
                generation_config=GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                )
            )
            
            # 5. Preparar la solicitud con contenido mixto (texto + imágenes)
            contents = [
                {"role": "user", "parts": [{"text": system_prompt}]},
            ]
            
            # Añadir imágenes al contenido
            parts = [{"text": human_prompt}]
            for i, frame in enumerate(frames):
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": frame
                    }
                })
            
            contents.append({"role": "user", "parts": parts})
            
            # 6. Generar respuesta
            response = model.generate_content(contents)
            
            # 7. Procesar la respuesta
            if not response.candidates or not response.candidates[0].content.parts:
                return {"error": "Gemini no generó una respuesta válida"}
            
            response_text = response.candidates[0].content.parts[0].text
            
            # 8. Intentar parsear JSON de la respuesta
            try:
                # Buscar JSON en la respuesta
                json_str = response_text
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                return {"texto": response_text, "error_json": "No se pudo parsear como JSON"}
        
        except Exception as e:
            print(f"Error en análisis de video con Gemini: {e}")
            traceback.print_exc()
            return {"error": f"Error en análisis: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Error en análisis: {e}")
        return {"error": str(e)}

def execute_gemini_ffmpeg_command(command_info: Dict[str, Any], input_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
    """
    Ejecuta un comando FFmpeg generado por Gemini
    
    Args:
        command_info: Información del comando a ejecutar (de Gemini)
        input_path: Ruta al video de entrada
        output_dir: Directorio para guardar la salida
        
    Returns:
        Tupla con (mensaje, ruta_al_video_procesado)
    """
    try:
        if not command_info or "comando" not in command_info:
            return "El comando FFmpeg no es válido", None
        
        # Obtener el comando
        cmd_str = command_info["comando"]
        
        # Asegurarse de que el comando tiene un archivo de salida
        if "-y" not in cmd_str:
            cmd_str += " -y"  # Agregar flag para sobrescribir
        
        # Crear nombre para archivo de salida si no está en el comando
        output_filename = f"processed_{os.path.basename(input_path)}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Asegurarnos de usar rutas absolutas para input_path y output_path
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        # Reemplazar input y output en el comando si es necesario
        if "INPUT_FILE" in cmd_str:
            cmd_str = cmd_str.replace("INPUT_FILE", input_path)
        elif "input.mp4" in cmd_str:
            cmd_str = cmd_str.replace("input.mp4", input_path)
        # Verificar nombres de archivo sin rutas
        elif os.path.basename(input_path) in cmd_str:
            cmd_str = cmd_str.replace(os.path.basename(input_path), input_path)
        
        if "OUTPUT_FILE" in cmd_str:
            cmd_str = cmd_str.replace("OUTPUT_FILE", output_path)
        elif "output.mp4" in cmd_str:
            cmd_str = cmd_str.replace("output.mp4", output_path)
        
        # Dividir el comando en una lista usando shlex para manejar espacios en rutas
        try:
            # En Windows, shlex.split no maneja bien las rutas con backslashes
            if os.name == 'nt':
                # Reemplazar backslashes con forward slashes para el parsing
                parsed_cmd = cmd_str.replace('\\', '/')
                cmd_parts = shlex.split(parsed_cmd)
                # Restaurar backslashes para los paths de Windows
                for i in range(len(cmd_parts)):
                    if '/' in cmd_parts[i] and (':/' in cmd_parts[i] or './' in cmd_parts[i]):
                        cmd_parts[i] = cmd_parts[i].replace('/', '\\')
            else:
                cmd_parts = shlex.split(cmd_str)
        except Exception as e:
            print(f"Error al parsear el comando: {e}")
            # Fallback al método original si hay error con shlex
            cmd_parts = cmd_str.split()
        
        # Asegurarse de que ffmpeg está al principio
        if len(cmd_parts) == 0 or cmd_parts[0] != "ffmpeg":
            cmd_parts.insert(0, "ffmpeg")
        
        # Eliminar comillas de los argumentos individuales
        for i in range(len(cmd_parts)):
            if cmd_parts[i].startswith('"') and cmd_parts[i].endswith('"'):
                cmd_parts[i] = cmd_parts[i][1:-1]
            elif cmd_parts[i].startswith("'") and cmd_parts[i].endswith("'"):
                cmd_parts[i] = cmd_parts[i][1:-1]
        
        # CORREGIR ORDEN DE PARÁMETROS: Asegurarse que el input se especifica correctamente
        # Crear una nueva lista de argumentos donde primero va 'ffmpeg', luego '-i', luego el archivo
        new_cmd_parts = ["ffmpeg"]
        input_specified = False
        input_index = -1
        
        # Buscar si ya hay un -i en el comando
        for i, part in enumerate(cmd_parts):
            if part == "-i" and i < len(cmd_parts) - 1:
                input_specified = True
                input_index = i
                break
        
        # Si no hay -i, añadirlo
        if not input_specified:
            new_cmd_parts.append("-i")
            new_cmd_parts.append(input_path)
        else:
            # Si hay un -i, pero no está justo después de 'ffmpeg', reordenar
            new_cmd_parts.append("-i")
            new_cmd_parts.append(cmd_parts[input_index + 1])  # El archivo de entrada
        
        # Añadir el resto de parámetros, omitiendo 'ffmpeg', '-i' y el archivo de entrada
        for i, part in enumerate(cmd_parts):
            if i == 0 and part == "ffmpeg":  # Omitir ffmpeg inicial
                continue
            if i == input_index or i == input_index + 1:  # Omitir -i y el path ya añadidos
                continue
            new_cmd_parts.append(part)
        
        # Usar los nuevos argumentos reordenados
        cmd_parts = new_cmd_parts
        
        # SOLUCIÓN PARA ERROR DE MEMORIA: 
        # Reducir la resolución del video para evitar el error de malloc
        resolution_added = False
        for i, part in enumerate(cmd_parts):
            if part == "-vf" and i < len(cmd_parts) - 1:
                # Si ya hay un filtro, añadir scale al principio
                cmd_parts[i+1] = f"scale=1280:-1,{cmd_parts[i+1]}"
                resolution_added = True
                break
        
        # Si no encontramos un filtro existente, agregar uno nuevo para escalar
        if not resolution_added:
            # Buscar la posición correcta para insertar (antes de output o codecs)
            insert_position = len(cmd_parts) - 1  # Por defecto al final
            for i, part in enumerate(cmd_parts):
                if part.startswith("-c:") or part == "-c" or part.endswith(".mp4") or part.endswith(".avi"):
                    insert_position = i
                    break
            
            # Insertar filtro de escala
            cmd_parts.insert(insert_position, "-vf")
            cmd_parts.insert(insert_position + 1, "scale=1280:-1")
        
        # Limitar threads para evitar problemas de memoria
        thread_specified = False
        for i, part in enumerate(cmd_parts):
            if part == "-threads" and i < len(cmd_parts) - 1:
                cmd_parts[i+1] = "4"  # Limitar a 4 threads
                thread_specified = True
                break
        
        # Si no se especificó threads, añadirlo
        if not thread_specified:
            cmd_parts.insert(len(cmd_parts) - 1, "-threads")
            cmd_parts.insert(len(cmd_parts) - 1, "4")
        
        # Imprimir el comando completo para debugging
        print(f"Ejecutando comando FFmpeg (modificado): {' '.join(cmd_parts)}")
        
        # Ejecutar el comando
        result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
        
        # Verificar si el archivo de salida existe
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return f"Comando ejecutado: {command_info.get('descripción', 'Operación FFmpeg')}", output_path
        else:
            # Intentar encontrar el archivo de salida en el comando
            for i, part in enumerate(cmd_parts):
                if part.endswith(".mp4") or part.endswith(".avi") or part.endswith(".mov"):
                    if i > 0 and cmd_parts[i-1] != "-i":  # No es un archivo de entrada
                        possible_output = part
                        if os.path.exists(possible_output) and os.path.getsize(possible_output) > 0:
                            return f"Comando ejecutado: {command_info.get('descripción')}", possible_output
            
            return "El comando se ejecutó pero no se encontró un archivo de salida válido", None
    
    except subprocess.CalledProcessError as e:
        # MODIFICACIÓN: Si tenemos error de x264 malloc, intentar de nuevo con resolución más baja
        if "malloc" in str(e.stderr) and "x264" in str(e.stderr):
            try:
                print("Error de memoria en x264. Reintentando con resolución muy reducida...")
                
                # Crear un comando más simple y con resolución muy baja
                simple_cmd = [
                    "ffmpeg", 
                    "-i", input_path,
                    "-vf", "scale=640:-1",  # Resolución mucho más baja
                    "-c:v", "libx264", 
                    "-preset", "ultrafast",  # Más rápido, menos memoria
                    "-crf", "30",  # Menor calidad para reducir carga
                    "-threads", "2",  # Menos threads
                    "-c:a", "copy",
                    "-y", output_path
                ]
                
                print(f"Reintentando con comando: {' '.join(simple_cmd)}")
                fallback_result = subprocess.run(simple_cmd, check=True, capture_output=True, text=True)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return f"Comando ejecutado con resolución reducida: {command_info.get('descripción', 'Operación FFmpeg')}", output_path
            except Exception as fallback_error:
                print(f"Error en el reintento: {fallback_error}")
        
        error_msg = f"Error ejecutando FFmpeg: {e.stderr}"
        print(error_msg)
        return error_msg, None
    except Exception as e:
        error_msg = f"Error general ejecutando comando: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None

# Función principal que engloba todo el proceso
def process_video_with_gemini(
    video_path: str,
    user_request: str,
    output_dir: str,
    api_key: Optional[str] = None
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """
    Procesa un video usando Gemini para análisis y generación de comandos FFmpeg
    
    Args:
        video_path: Ruta al video de entrada
        user_request: Solicitud del usuario en lenguaje natural
        output_dir: Directorio para guardar los archivos de salida
        api_key: API key de Google Gemini (opcional)
        
    Returns:
        Tupla con (mensaje, ruta_al_video_procesado, respuesta_completa_de_gemini)
    """
    try:
        # 1. Analizar el video con Gemini
        gemini_response = analyze_video_with_gemini(
            video_path=video_path,
            user_prompt=user_request,
            api_key=api_key,
            temp_dir=output_dir
        )
        
        # 2. Verificar si hubo error en el análisis
        if "error" in gemini_response:
            return f"Error en análisis con Gemini: {gemini_response['error']}", None, gemini_response
        
        # 3. Extraer y ejecutar los comandos FFmpeg
        if "comandos_ffmpeg" in gemini_response and gemini_response["comandos_ffmpeg"]:
            # Tomar el primer comando (podría haber varios)
            comando = gemini_response["comandos_ffmpeg"][0]
            
            # Ejecutar el comando
            message, output_path = execute_gemini_ffmpeg_command(
                comando, video_path, output_dir
            )
            
            if output_path:
                return message, output_path, gemini_response
            else:
                return f"Error ejecutando comando: {message}", None, gemini_response
        else:
            return "Gemini no generó comandos FFmpeg para esta solicitud", None, gemini_response
    
    except Exception as e:
        error_msg = f"Error en process_video_with_gemini: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None, {"error": error_msg}

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .video_processor import extract_frames, get_video_info

def process_video_with_gemini(video_path: str, interval_secs: int = 5) -> Dict[str, Any]:
    """Procesa el video y extrae frames para análisis"""
    try:
        # Extraer frames
        frames = extract_frames(video_path, interval_secs)
        if not frames:
            return {"error": "No se pudieron extraer frames"}
            
        # Obtener info del video
        video_info = get_video_info(video_path)
        
        return {
            "frames": frames,
            "video_info": video_info
        }
    except Exception as e:
        return {"error": f"Error procesando video: {str(e)}"}

def analyze_video_with_gemini(frames: List[str], prompt: str = "") -> Dict[str, Any]:
    """Analiza los frames con Gemini"""
    try:
        # Configurar Gemini
        if "GOOGLE_API_KEY" in os.environ:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        # Crear modelo
        model = genai.GenerativeModel("gemini-pro-vision")
        
        # Crear modelo
        model = genai.GenerativeModel("gemini-pro-vision")
        
        # Preparar contenido
        contents = []
        for frame in frames:
            contents.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": frame
                }
            })
            
        # Generar análisis
        response = model.generate_content([
            {"text": prompt or "Analiza este video y describe lo que ves"},
            *contents
        ])
        
        return {
            "análisis": response.text,
            "frames_analizados": len(frames)
        }
        
    except Exception as e:
        return {"error": f"Error en análisis: {str(e)}"}
