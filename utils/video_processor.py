import os
import time
import tempfile
import subprocess # Primary tool now
import shutil
import json
import shlex # For safe command line argument splitting
from typing import Tuple, Dict, List, Any, Optional
from moviepy.editor import VideoFileClip
from PIL import Image
import io
import base64
import numpy as np

# Still need pydub for silence detection
from pydub import AudioSegment, silence

# --- FFmpeg Helper Functions ---

def run_ffmpeg_command(cmd_list: List[str], operation_desc: str = "FFmpeg operation") -> None:
    """Runs an FFmpeg command using subprocess, raising an error on failure."""
    print(f"Running {operation_desc}: {' '.join(shlex.quote(str(arg)) for arg in cmd_list)}")
    try:
        # Set startupinfo for Windows to hide the console window
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True,
                                encoding='utf-8', errors='ignore', startupinfo=startupinfo)
        print(f"{operation_desc} successful.")
        # print(f"FFmpeg STDERR:\n{result.stderr}") # Uncomment for detailed ffmpeg logs
    except subprocess.CalledProcessError as e:
        error_message = f"""Error during {operation_desc} (Return Code: {e.returncode}):
Command: {' '.join(shlex.quote(str(arg)) for arg in cmd_list)}
STDOUT:
{e.stdout}
STDERR:
{e.stderr}"""
        print(error_message)
        raise Exception(f"{operation_desc} failed. Check FFmpeg logs above.")
    except FileNotFoundError:
        raise FileNotFoundError("FFmpeg command not found. Make sure FFmpeg is installed and in your system PATH.")
    except Exception as e:
        import traceback
        print(f"Unexpected error running {operation_desc}:")
        traceback.print_exc()
        raise Exception(f"Unexpected error during {operation_desc}: {e}")


def build_ffmpeg_filters(operations: List[Dict[str, Any]], video_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Builds FFmpeg -vf (video filter) and -af (audio filter) strings from operations."""
    vf_parts = []
    af_parts = []
    speed_factor = 1.0 # Track overall speed for audio adjustment

    for op in operations:
        op_type = op.get("type")
        params = op.get("params", {})

        if op_type == "speed":
            factor = params.get("factor", 1.0)
            if abs(factor - 1.0) > 1e-6:
                if factor <= 0: factor = 0.01 # Avoid invalid
                speed_factor *= factor # Accumulate speed changes
                # Note: setpts handles video speed. Audio speed needs atempo below.
                # We apply setpts at the end based on the *final* speed_factor
            
        elif op_type == "color_adjustment":
            brightness = params.get("brightness", 1.0)
            contrast = params.get("contrast", 1.0)
            saturation = params.get("saturation", 1.0)
            color_changed = abs(brightness - 1.0) > 1e-6 or abs(contrast - 1.0) > 1e-6 or abs(saturation - 1.0) > 1e-6
            if color_changed:
                ffmpeg_brightness = brightness - 1.0
                ffmpeg_contrast = contrast
                ffmpeg_saturation = saturation
                vf_parts.append(f"eq=brightness={ffmpeg_brightness:.4f}:contrast={ffmpeg_contrast:.4f}:saturation={ffmpeg_saturation:.4f}")
            
        elif op_type == "add_text":
            text = params.get("text", "")
            if text:
                font_size = params.get("font_size", 30)
                position = params.get("position", "center")
                color = params.get("color", "white")
                # Font handling is OS-dependent. This path needs to be correct for your system.
                # Common Windows path: C:/Windows/Fonts/arial.ttf
                # Common Linux path: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
                font_file = "C:/Windows/Fonts/arial.ttf" # Adjust if necessary!
                if not os.path.exists(font_file):
                     # Fallback or error if font not found
                     print(f"Warning: Font file '{font_file}' not found. Text may not render correctly.")
                     # Provide a very basic fallback font if possible or skip
                     font_file = "Arial" # Relying on system mapping (might not work)


                x_pos, y_pos = "(w-text_w)/2", "(h-text_h)/2" # Default center using FFmpeg expressions
                if position == "top": y_pos = "50"
                elif position == "bottom": y_pos = "h-th-50"
                # Escape special characters for FFmpeg filter syntax
                escaped_text = shlex.quote(text).strip("'") # More robust escaping

                vf_parts.append(
                    f"drawtext=fontfile='{font_file}':text='{escaped_text}':"
                    f"fontsize={font_size}:fontcolor={color}:x='{x_pos}':y='{y_pos}'"
                )
        elif op_type == "crop":
            x1 = params.get("x1", 0)
            y1 = params.get("y1", 0)
            x2 = params.get("x2", video_info.get("size",[0,0])[0])
            y2 = params.get("y2", video_info.get("size",[0,0])[1])
            w = max(1, x2-x1) # Ensure width is positive
            h = max(1, y2-y1) # Ensure height is positive
            vf_parts.append(f"crop={w}:{h}:{x1}:{y1}")
        elif op_type == "rotate":
             angle = params.get("angle", 0)
             if angle != 0:
                  radians = angle * (3.1415926535 / 180.0)
                  # Use transpose filter for 90/180/270 degree rotations (often faster/better)
                  if angle == 90: vf_parts.append("transpose=1")
                  elif angle == 180: vf_parts.append("transpose=2,transpose=2") # Rotate 180
                  elif angle == 270 or angle == -90: vf_parts.append("transpose=2")
                  else: # Use rotate filter for arbitrary angles
                      vf_parts.append(f"rotate={radians}:ow=rotw({radians}):oh=roth({radians})")
        # --- Add mappings for other simple FFmpeg filters here ---

    # --- Apply final speed adjustments ---
    if abs(speed_factor - 1.0) > 1e-6:
         if speed_factor <= 0: speed_factor = 0.01
         vf_parts.insert(0, f"setpts=PTS/{speed_factor}") # Apply setpts first
         atempo_filter_str = generate_atempo_filter(speed_factor)
         if atempo_filter_str:
              af_parts.append(atempo_filter_str)

    return vf_parts, af_parts

def generate_atempo_filter(factor):
    """Generates FFmpeg atempo filter string, handling factors outside 0.5-100."""
    if factor <= 0: return "atempo=0.5"
    filters = []
    current_factor = factor
    while current_factor < 0.5:
        filters.append("atempo=0.5")
        current_factor /= 0.5
        if len(filters) > 10: print("Warning: Speed factor too low"); break
    while current_factor > 2.0:
         filters.append("atempo=2.0")
         current_factor /= 2.0
         if len(filters) > 20: print("Warning: Speed factor too high"); break
    if abs(current_factor - 1.0) > 1e-6:
         final_tempo = max(0.5, min(current_factor, 100.0))
         filters.append(f"atempo={final_tempo:.6f}")
    return ",".join(filters) if filters else ""

# --- Main Processing Function ---

def process_video(input_path: str, operations: List[Dict[str, Any]], output_dir: str) -> Optional[str]:
    """
    Process a video with a series of operations using FFmpeg
    
    Args:
        input_path: Path to the input video
        operations: List of operations to perform
        output_dir: Directory to save the output video
        
    Returns:
        Path to the processed video, or None if processing failed
    """
    print("Applying filter operations...")
    current_input = input_path
    
    # Create timestamp for unique filenames
    timestamp = str(int(time.time()))
    
    # Process each operation in sequence
    for i, operation in enumerate(operations):
        operation_type = operation.get('type')
        params = operation.get('params', {})
        
        # Skip empty or unknown operations
        if not operation_type:
            continue
            
        # Create output path for this operation
        output_path = os.path.join(output_dir, f"filtered_{timestamp}.mp4")
        
        # Apply the operation based on its type
        if operation_type == 'trim':
            start_time = params.get('start_time', 0)
            end_time = params.get('end_time')
            
            if end_time is None:
                # Get video duration if end_time is not specified
                probe = ffmpeg.probe(current_input)
                duration = float(probe['format']['duration'])
                end_time = duration
            
            # Construct ffmpeg command
            cmd = [
                "ffmpeg", "-i", current_input,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-y", output_path
            ]
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Update current_input for next operation
            current_input = output_path
            
        elif operation_type == 'speed':
            factor = params.get('factor', 1.0)
            
            # Separate audio and video speed adjustment for better quality
            cmd = [
                "ffmpeg", "-i", current_input,
                "-filter_complex", f"[0:v]setpts={1/factor}*PTS[v];[0:a]atempo={min(2.0, factor)}[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-y", output_path
            ]
            
            # For extreme speed changes, we might need multiple atempo filters
            if factor > 2.0:
                atempo_chain = ""
                remaining_factor = factor
                while remaining_factor > 1.0:
                    current_factor = min(2.0, remaining_factor)
                    atempo_chain += f"atempo={current_factor},"
                    remaining_factor /= current_factor
                
                # Remove trailing comma
                atempo_chain = atempo_chain[:-1]
                
                # Update command with the atempo chain
                cmd = [
                    "ffmpeg", "-i", current_input,
                    "-filter_complex", f"[0:v]setpts={1/factor}*PTS[v];[0:a]{atempo_chain}[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    "-y", output_path
                ]
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Update current_input for next operation
            current_input = output_path
            
        elif operation_type == 'cut_silences':
            min_silence_len = params.get('min_silence_len', 500)  # in ms
            silence_thresh = params.get('silence_thresh', -40)  # in dB
            
            # Use the separate function for cutting silences
            try:
                new_path = cut_silences(
                    current_input, 
                    output_dir,
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_thresh
                )
                if new_path:
                    current_input = new_path
            except Exception as e:
                print(f"Error cutting silences: {e}")
                # Continue to next operation
                
        elif operation_type == 'color_adjustment':
            brightness = params.get('brightness', 1.0)
            contrast = params.get('contrast', 1.0)
            saturation = params.get('saturation', 1.0)
            
            # Construct the eq filter parameters
            eq_params = f"eq=brightness={brightness:.4f}:contrast={contrast:.4f}:saturation={saturation:.4f}"
            
            # Construct the ffmpeg command
            cmd = [
                "ffmpeg", "-i", current_input,
                "-vf", eq_params,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-y", output_path
            ]
            
            print(f"Running Applying filters: {' '.join(cmd)}")
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            print("Applying filters successful.")
            
            # Update current_input for next operation
            current_input = output_path
            
        elif operation_type == 'add_text':
            text = params.get('text', 'Sample Text')
            position = params.get('position', 'center')
            font_size = params.get('font_size', 24)
            color = params.get('color', 'white')
            
            # Map position to coordinates
            position_map = {
                'top': "x=(w-text_w)/2:y=h*0.1",
                'center': "x=(w-text_w)/2:y=(h-text_h)/2",
                'bottom': "x=(w-text_w)/2:y=h*0.9"
            }
            
            pos = position_map.get(position, position_map['center'])
            
            # Escape special characters in the text
            escaped_text = text.replace("'", "\\'").replace(":", "\\:")
            
            # Construct the drawtext filter
            filter_text = f"drawtext=text='{escaped_text}':fontsize={font_size}:fontcolor={color}:{pos}"
            
            # Construct the ffmpeg command
            cmd = [
                "ffmpeg", "-i", current_input,
                "-vf", filter_text,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "copy",
                "-y", output_path
            ]
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Update current_input for next operation
            current_input = output_path
            
        elif operation_type == 'crop':
            # Get crop parameters
            width = params.get('width')
            height = params.get('height')
            x = params.get('x', 0)
            y = params.get('y', 0)
            
            # Get video dimensions if needed
            if width is None or height is None:
                probe = ffmpeg.probe(current_input)
                video_info = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
                if video_info:
                    video_width = int(video_info['width'])
                    video_height = int(video_info['height'])
                    width = width or video_width
                    height = height or video_height
                else:
                    # Skip if can't determine video dimensions
                    continue
            
            # Construct the crop filter
            filter_crop = f"crop={width}:{height}:{x}:{y}"
            
            # Construct the ffmpeg command
            cmd = [
                "ffmpeg", "-i", current_input,
                "-vf", filter_crop,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "copy",
                "-y", output_path
            ]
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Update current_input for next operation
            current_input = output_path
            
        elif operation_type == 'black_and_white':
            # Construct the ffmpeg command to convert to black and white
            cmd = [
                "ffmpeg", "-i", current_input,
                "-vf", "format=gray",
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "copy",
                "-y", output_path
            ]
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Update current_input for next operation
            current_input = output_path
        
        elif operation_type == 'rotate':
            angle = params.get('angle', 0)
            
            # Map common angles to FFmpeg rotate filters
            if angle == 90:
                filter_rotate = "transpose=1"  # 90 degrees clockwise
            elif angle == 180:
                filter_rotate = "transpose=2,transpose=2"  # 180 degrees
            elif angle == 270 or angle == -90:
                filter_rotate = "transpose=2"  # 90 degrees counterclockwise
            else:
                # For arbitrary angles (less efficient)
                filter_rotate = f"rotate={angle}*PI/180"
            
            # Construct the ffmpeg command
            cmd = [
                "ffmpeg", "-i", current_input,
                "-vf", filter_rotate,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "copy",
                "-y", output_path
            ]
            
            # Run the command
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Update current_input for next operation
            current_input = output_path
    
    # Finalize the video (copy to final output path)
    final_output_path = os.path.join(output_dir, f"final_output_{timestamp}.mp4")
    shutil.copy2(current_input, final_output_path)
    
    print(f"Processing complete. Final video: {final_output_path}")
    return final_output_path

# --- Silence Cutting (using pydub + ffmpeg) ---
def cut_silences(video_path: str, min_silence_len: int = 500, silence_thresh: int = -40, output_dir: Optional[str] = None, target_output_path: Optional[str] = None) -> str:
    """Detects and cuts silent parts using pydub for detection and FFmpeg for extraction/concatenation."""
    if output_dir is None: raise ValueError("output_dir must be provided for cut_silences temp files.")
    if not os.path.exists(video_path): raise FileNotFoundError(f"Input video for cut_silences not found: {video_path}")

    temp_suffix = str(int(time.time()))
    output_path = target_output_path or os.path.join(output_dir, f"output_no_silence_{temp_suffix}.mp4")
    temp_audio = os.path.join(output_dir, f"temp_audio_silence_{temp_suffix}.wav")
    temp_concat_list = os.path.join(output_dir, f"concat_list_{temp_suffix}.txt")
    segment_files = [] # Keep track of generated segment files
    cleanup_files = [temp_audio, temp_concat_list] # Add main temp files

    try:
        print(f"Cutting silences: min_len={min_silence_len}ms, thresh={silence_thresh}dB from {video_path}")

        # 1. Extract audio using FFmpeg
        cmd_extract_audio = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-y", temp_audio]
        run_ffmpeg_command(cmd_extract_audio, "Extracting audio for silence detection")

        # 2. Detect non-silent parts using pydub
        if not os.path.exists(temp_audio): raise RuntimeError(f"Temporary audio file not created: {temp_audio}")
        try:
            audio = AudioSegment.from_file(temp_audio)
        except Exception as pydub_err:
            raise Exception(f"Pydub failed to load extracted audio '{temp_audio}': {pydub_err}")

        print("Detecting non-silent parts...")
        non_silent_parts_ms = silence.detect_nonsilent(
            audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=1
        )

        if not non_silent_parts_ms:
            print("No non-silent parts detected. Returning original video path (or copy).")
            if target_output_path and video_path != target_output_path:
                 shutil.copy2(video_path, target_output_path)
                 return target_output_path
            return video_path
        
        non_silent_parts_sec = [(start / 1000.0, end / 1000.0) for start, end in non_silent_parts_ms]
        print(f"Found {len(non_silent_parts_sec)} non-silent segments.")

        # 3. Extract non-silent video segments using FFmpeg and create concat list
        with open(temp_concat_list, 'w', encoding='utf-8') as f:
            for i, (start, end) in enumerate(non_silent_parts_sec):
                if end - start > 0.01: # Avoid zero duration segments
                    # Ensure segment filenames are safe, especially on Windows
                    segment_basename = f"segment_{temp_suffix}_{i}.mp4"
                    segment_filename = os.path.join(output_dir, segment_basename)
                    segment_files.append(segment_filename)
                    cleanup_files.append(segment_filename) # Add segment to cleanup list
                    # Use relative path for concat file, ensure proper formatting
                    f.write(f"file '{segment_basename}'\n") # Use forward slashes for paths in concat file?

                    cmd_extract_segment = [
                        "ffmpeg", "-i", video_path, "-ss", str(start), "-to", str(end),
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero",
                        "-y", segment_filename
                    ]
                    run_ffmpeg_command(cmd_extract_segment, f"Extracting segment {i+1}")
                else:
                     print(f"Skipping segment {i+1} due to short duration ({start=}, {end=})")

        # 4. Concatenate segments using FFmpeg
        cmd_concat = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", temp_concat_list, "-c", "copy", "-y", output_path]
        run_ffmpeg_command(cmd_concat, "Concatenating segments")

        return output_path
    
    except Exception as e:
        import traceback
        print("--- Error during Silence Cutting ---")
        traceback.print_exc()
        print("----------------------------------")
        # Return original path on error? Or raise? Raising is safer.
        # shutil.copy2(video_path, temp_output) # Option: copy original to output on failure
        # output_path_to_return = temp_output
        raise Exception(f"Error cutting silences: {str(e)}")
    finally:
        # Ensure all clips are closed
        if segment_files:
            for f in segment_files:
                 try:
                     print(f"Cleaning up segment file: {f}")
                     os.remove(f)
                 except OSError as e:
                     print(f"Warning: Could not remove segment file {f}: {e}")
        if temp_audio in cleanup_files and os.path.exists(temp_audio):
            try:
                print(f"Cleaning up temporary audio file: {temp_audio}")
                os.remove(temp_audio)
            except OSError as e:
                print(f"Warning: Could not remove temporary audio file {temp_audio}: {e}")
        if temp_concat_list in cleanup_files and os.path.exists(temp_concat_list):
            try:
                print(f"Cleaning up temporary concat list file: {temp_concat_list}")
                os.remove(temp_concat_list)
            except OSError as e:
                print(f"Warning: Could not remove temporary concat list file {temp_concat_list}: {e}")

    return output_path

# --- Info and Thumbnail Functions (using ffprobe/ffmpeg) ---

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Gets video information using ffprobe. No Moviepy fallback."""
    if not os.path.exists(video_path): return {"error": "Video file not found"}
    try:
        # Verify ffprobe exists
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
        data = json.loads(result.stdout)

        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if not video_stream: return {"error": "No video stream found."}

        duration_str = data.get("format", {}).get("duration") or video_stream.get("duration")
        duration = float(duration_str) if duration_str is not None else 0.0
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        fps_str = video_stream.get("avg_frame_rate", "0/1")
        try: num, den = map(int, fps_str.split('/')); fps = float(num)/float(den) if den else 0.0
        except: fps = 0.0
        rotation = 0
        try: rotation = int(video_stream.get('tags', {}).get('rotate', 0))
        except ValueError: rotation = 0

        if abs(rotation) in [90, 270]: width, height = height, width

        return {
                    "duration": duration,
            "size": [width, height],
            "fps": fps if fps > 0 else 30.0, # Default FPS if calculation fails
            "format": data.get("format", {}).get("format_name", "unknown"),
            "codec_video": video_stream.get("codec_name", "unknown"),
            "codec_audio": audio_stream.get("codec_name", "unknown") if audio_stream else None,
            "audio": audio_stream is not None,
            "rotation": rotation
        }
    except subprocess.CalledProcessError as ffprobe_err:
        # More specific error message for ffprobe failure
        print(f"ffprobe failed ({ffprobe_err}). Ensure ffprobe is installed and accessible.")
        return {"error": f"ffprobe execution failed: {ffprobe_err}"}
    except FileNotFoundError:
        return {"error": "ffprobe command not found. Ensure FFmpeg is installed and in PATH."}
    except Exception as e:
        print(f"Error getting video info via ffprobe: {e}")
        return {"error": f"Failed to get video info: {e}"}

def get_video_thumbnail(video_path: str, time_sec: float = 1.0, output_dir: Optional[str] = None) -> Optional[str]:
    """Extracts a thumbnail using FFmpeg. No Moviepy fallback."""
    if not os.path.exists(video_path): return None
    if output_dir is None: output_dir = tempfile.gettempdir()

    thumb_suffix = str(int(time.time())) + "_" + str(hash(video_path))
    thumbnail_path = os.path.join(output_dir, f"thumbnail_{thumb_suffix}.jpg")

    # Get duration safely, defaulting if info fails
    duration = 0.0
    try:
        info = get_video_info(video_path)
        duration = info.get("duration", 0.0)
        if "error" in info: duration = 0.0 # Reset if info retrieval had an error
    except Exception:
        pass # Use duration = 0.0

    if duration <= 0: time_sec = 0.0
    else: time_sec = max(0.0, min(time_sec, duration - 0.1 if duration > 0.1 else 0.0))

    try:
        cmd = [
            "ffmpeg",
            "-ss", str(time_sec),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "3",
            "-vf", "scale=320:-1",
            "-y",
            thumbnail_path
        ]
        run_ffmpeg_command(cmd, "Generating thumbnail")
        return thumbnail_path if os.path.exists(thumbnail_path) else None
    except Exception as e:
        print(f"Error generating thumbnail with FFmpeg: {e}")
        # REMOVED Moviepy fallback logic
        return None

def extract_frames(video_path: str, interval_secs: int = 5) -> List[str]:
    """Extrae frames del video cada N segundos"""
    frames = []
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        for t in range(0, int(duration), interval_secs):
            frame = clip.get_frame(t)
            # Convertir a PIL y redimensionar
            img = Image.fromarray(frame)
            img = img.resize((640, 360))
            # Convertir a base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            frames.append(img_str)
    return frames

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Obtiene metadata del video"""
    with VideoFileClip(video_path) as clip:
        return {
            "duration": clip.duration,
            "width": clip.w,
            "height": clip.h,
            "fps": clip.fps
        }
