import os
import re
import json
import time
import traceback
import streamlit as st
import subprocess
from typing import List, Dict, Any, Tuple, Optional
from utils.video_processor import process_video, cut_silences
from utils.ai_integrations import (
    process_command_with_openai, 
    process_command_with_anthropic,
)

def parse_command(user_command: str, ai_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse a natural language command and the AI response into executable operations
    
    Args:
        user_command: The user's original command
        ai_response: The AI's response with operations to perform
        
    Returns:
        List of operations to perform
    """
    print(f"DEBUG: parse_command called (likely unused if AI provides operations directly). AI Response: {ai_response}")
    # Check if there's an error in the AI response
    if "error" in ai_response:
        error_message = ai_response.get("error", "Unknown error")
        raise Exception(f"AI Error: {error_message}")
    
    # Extract operations from the AI response
    operations = ai_response.get("operations", [])
    
    # If no operations were returned, DO NOT fall back for now to simplify debugging
    # if not operations:
    #     print("DEBUG: AI returned no operations, attempting fallback parser.") # Add log here too
    #     operations = fallback_command_parser(user_command)
    
    # Cache for already processed operation types to speed up normalization
    operation_type_cache = {}
    
    # Validate that each operation has the required fields - optimized validation
    for operation in operations:
        # Use dict.get with default instead of checking and setting
        operation_type = operation.get("type")
        if not operation_type:
            raise Exception("Invalid operation: 'type' field is required")
        
        # Add default params if missing
        if "params" not in operation:
            operation["params"] = {}
        
        # Add default timeline_position if missing
        if "timeline_position" not in operation:
            operation["timeline_position"] = "all"
        
        # Normalize operation types using cache for speed
        if operation_type in operation_type_cache:
            operation["type"] = operation_type_cache[operation_type]
        elif operation_type == "color_adjust":
            operation["type"] = "color_adjustment"
            operation_type_cache["color_adjust"] = "color_adjustment"
    
    return operations

# Comment out the fallback function entirely for now
'''
def fallback_command_parser(command: str) -> List[Dict[str, Any]]:
    """
    Fallback parser for simple commands when AI fails
    (COMMENTED OUT FOR DEBUGGING)
    """
    operations = []
    
    # Check for trim command
    trim_match = re.search(r'trim.*?(\d+\.?\d*).*?(\d+\.?\d*)', command.lower())
    if trim_match:
        start_time = float(trim_match.group(1))
        end_time = float(trim_match.group(2))
        operations.append({
            "type": "trim",
            "params": {
                "start_time": start_time,
                "end_time": end_time
            },
            "timeline_position": "all"
        })
    
    # Check for cut silences command
    if re.search(r'(cut|remove).*?(silence|silent)', command.lower()):
        operations.append({
            "type": "cut_silences",
            "params": {
                "min_silence_len": 500,
                "silence_thresh": -40
            },
            "timeline_position": "all"
        })
    
    # Check for speed command
    speed_match = re.search(r'(speed|faster|slower).*?(\d+\.?\d*)', command.lower())
    if speed_match:
        factor = float(speed_match.group(2))
        # Adjust if value is likely a percentage
        if factor > 10:
            factor = factor / 100
        operations.append({
            "type": "speed",
            "params": {
                "factor": factor
            },
            "timeline_position": "all"
        })
    
    # Check for color adjustment
    if re.search(r'(color|brightness|contrast|saturation)', command.lower()):
        brightness = 1.0
        contrast = 1.0
        saturation = 1.0
        
        brightness_match = re.search(r'brightness.*?(\d+\.?\d*)', command.lower())
        if brightness_match:
            brightness = float(brightness_match.group(1))
            if brightness > 10:  # Likely a percentage
                brightness = brightness / 100
        
        contrast_match = re.search(r'contrast.*?(\d+\.?\d*)', command.lower())
        if contrast_match:
            contrast = float(contrast_match.group(1))
            if contrast > 10:  # Likely a percentage
                contrast = contrast / 100
        
        saturation_match = re.search(r'saturation.*?(\d+\.?\d*)', command.lower())
        if saturation_match:
            saturation = float(saturation_match.group(1))
            if saturation > 10:  # Likely a percentage
                saturation = saturation / 100
        
        operations.append({
            "type": "color_adjustment",
            "params": {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation
            },
            "timeline_position": "all"
        })
    
    # Check for add text
    text_match = re.search(r'add.*?text.*?["\'](.+?)["\']', command.lower())
    if text_match:
        text = text_match.group(1)
        position = "center"
        
        if "top" in command.lower():
            position = "top"
        elif "bottom" in command.lower():
            position = "bottom"
        
        operations.append({
            "type": "add_text",
            "params": {
                "text": text,
                "position": position,
                "font_size": 30,
                "color": "white"
            },
            "timeline_position": "all"
        })
    
    # Return the list of operations (may be empty if no patterns matched)
    return operations
'''

def execute_command(operations: List[Dict[str, Any]], input_video_path: str, output_dir: str) -> Tuple[str, Optional[str]]:
    """
    Ejecuta una lista de operaciones de edición en un video.
    Actualmente, simplificado para llamar a process_video con todas las operaciones.
    """
    print(f"DEBUG: execute_command received {len(operations)} operations for video {input_video_path}")
    st.write(f"DEBUG: execute_command received operations: {operations}") # Show in UI too

    if not operations:
        return "No operations to execute.", input_video_path # Return original if no ops
    
    if not input_video_path or not os.path.exists(input_video_path):
        raise ValueError(f"Input video path is invalid or file does not exist: {input_video_path}")

    try:
        # Verificar si hay operaciones de tipo gemini_ffmpeg (comandos directos de FFmpeg)
        gemini_ffmpeg_ops = [op for op in operations if op.get('type') == 'gemini_ffmpeg']
        if gemini_ffmpeg_ops:
            print("DEBUG: Found gemini_ffmpeg operation, executing FFmpeg command directly")
            
            # Obtener la operación (usar la primera si hay varias)
            op = gemini_ffmpeg_ops[0]
            params = op.get('params', {})
            
            # Verificar que tenemos los campos necesarios
            if 'comando' not in params:
                raise ValueError("La operación gemini_ffmpeg no contiene el comando FFmpeg necesario")
            
            # Crear nombre para el archivo de salida
            output_filename = f"gemini_ffmpeg_{os.path.basename(input_video_path)}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Extraer el comando y prepararlo
            cmd_str = params['comando']
            
            # Asegurarse de que el comando tiene un archivo de salida
            if "-y" not in cmd_str:
                cmd_str += " -y"  # Agregar flag para sobrescribir
            
            # Reemplazar rutas de entrada y salida en el comando
            if "INPUT_FILE" in cmd_str:
                cmd_str = cmd_str.replace("INPUT_FILE", input_video_path)
            elif "input.mp4" in cmd_str:
                cmd_str = cmd_str.replace("input.mp4", input_video_path)
            
            if "OUTPUT_FILE" in cmd_str:
                cmd_str = cmd_str.replace("OUTPUT_FILE", output_path)
            elif "output.mp4" in cmd_str:
                cmd_str = cmd_str.replace("output.mp4", output_path)
            
            # Dividir el comando en una lista para subprocess, manteniendo las partes con comillas juntas
            import shlex
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
            
            # Ejecutar el comando
            print(f"DEBUG: Ejecutando comando FFmpeg: {' '.join(cmd_parts)}")
            try:
                # Set startupinfo for Windows to hide the console window
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                
                result = subprocess.run(cmd_parts, check=True, capture_output=True, text=True, 
                                      encoding='utf-8', errors='ignore', startupinfo=startupinfo)
                
                # Verificar que la salida existe
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return f"Ejecutado: {params.get('descripción', 'Comando FFmpeg de Gemini')}", output_path
                else:
                    # Intentar encontrar un archivo de salida alternativo
                    for i, part in enumerate(cmd_parts):
                        if part.endswith(".mp4") or part.endswith(".avi") or part.endswith(".mov"):
                            if i > 0 and cmd_parts[i-1] != "-i":  # No es archivo de entrada
                                possible_output = part
                                if os.path.exists(possible_output) and os.path.getsize(possible_output) > 0:
                                    return f"Ejecutado: {params.get('descripción')}", possible_output
                    
                    raise ValueError("El comando FFmpeg se ejecutó pero no se encontró un archivo de salida válido")
            
            except subprocess.CalledProcessError as e:
                error_msg = f"Error ejecutando FFmpeg: {e.stderr}"
                print(error_msg)
                raise ValueError(error_msg)
        
        # Si no hay operaciones de gemini_ffmpeg, usar el process_video normal
        else:
            # Asumiendo que process_video puede manejar la lista de operaciones
            print("DEBUG: Calling process_video...")
            new_video_path = process_video(input_video_path, operations, output_dir)
            
            print(f"DEBUG: process_video returned: {new_video_path}")
            if new_video_path and os.path.exists(new_video_path) and new_video_path != input_video_path:
                 message = f"Video procesado con {len(operations)} operaciones."
                 return message, new_video_path
            elif new_video_path == input_video_path:
                 # This might happen if operations resulted in no actual change
                 message = "Las operaciones solicitadas no modificaron el video."
                 return message, input_video_path
            else:
                 # This case indicates an issue within process_video if it didn't return a valid path or returned None
                 print(f"ERROR: process_video did not return a valid existing path: {new_video_path}")
                 raise Exception("El procesamiento del video falló internamente (process_video no devolvió una ruta válida o existente). Verifique los logs de FFmpeg en la terminal.")

    except Exception as e:
        print(f"ERROR in execute_command: {e}")
        traceback.print_exc() # Print detailed traceback to terminal
        # Re-raise the exception so it propagates to the UI via on_command_complete
        raise Exception(f"Error al ejecutar el comando de edición: {str(e)}")

def extract_timeline_modification(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract timeline modification information from operations
    
    Args:
        operations: List of operations
        
    Returns:
        Timeline modification data
    """
    # Initialize with empty operations
    timeline_data = {
        "operations": []
    }
    
    # Filter operations that modify the timeline
    timeline_ops = []
    for operation in operations:
        op_type = operation.get("type", "")
        
        # Check if this operation type affects the timeline
        if op_type in ["trim", "add_marker", "add_effect", "split"]:
            timeline_ops.append(operation)
    
    # Add the timeline operations
    timeline_data["operations"] = timeline_ops
    
    return timeline_data
