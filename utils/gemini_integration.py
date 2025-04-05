"""
Integration with Google's Gemini 2.0 Flash API for fast command processing
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
import google.generativeai as genai

def setup_gemini_api():
    """Setup the Gemini API with the API key from environment variables"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return True

def process_command_with_gemini_flash(command: str, timeline_data: Any) -> Dict[str, Any]:
    """
    Process a video editing command using Gemini 2.0 Flash for fastest response times
    
    Args:
        command: The natural language command from the user
        timeline_data: The current timeline data to consider for context
        
    Returns:
        Parsed command response from Gemini
    """
    try:
        # Prepare timeline data as string for context
        if hasattr(timeline_data, 'get_serializable_data'):
            timeline_context = json.dumps(timeline_data.get_serializable_data())
        else:
            timeline_context = json.dumps(timeline_data)
        
        # Create the prompt optimized for speed and efficiency
        prompt = f"""
        You are a professional video editing assistant. Translate natural language commands into video editing operations.
        
        COMMAND: {command}
        
        CURRENT TIMELINE DATA: {timeline_context}
        
        Return a valid JSON with:
        1. "operations": A list of operations to perform
        2. "explanation": A brief explanation of what will be done
        
        Each operation must include:
        - "type": the operation type (cut_silence, color_adjust, trim, etc.)
        - "params": parameters specific to that operation
        - "timeline_position": where to apply the change ("all", "start", "end", or time ranges)
        """
        
        # Try to use the newest Gemini model first, with fallbacks
        try:
            # Use Gemini Flash if available (for fastest response time)
            model = genai.GenerativeModel('gemini-1.5-flash',
                                        generation_config={
                                            "temperature": 0.0,
                                            "max_output_tokens": 1024
                                        })
            response = model.generate_content(prompt)
        except Exception as model_error:
            # Fallback to other models if Flash not available
            try:
                model = genai.GenerativeModel('gemini-1.5-pro',
                                            generation_config={
                                                "temperature": 0.0,
                                                "max_output_tokens": 1024
                                            })
                response = model.generate_content(prompt)
            except:
                try:
                    model = genai.GenerativeModel('gemini-pro',
                                                generation_config={
                                                    "temperature": 0.0,
                                                    "max_output_tokens": 1024
                                                })
                    response = model.generate_content(prompt)
                except:
                    raise Exception(f"No Gemini model available: {str(model_error)}")
        
        # Extract and parse the JSON response
        response_text = response.text
        
        # Handle potential markdown formatting in response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text
        
        result = json.loads(json_str)
        return result
    
    except Exception as e:
        return {"error": f"Error processing command with Gemini: {str(e)}"}

def analyze_video_with_gemini(frames: List[str], audio_transcript: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze video content using Gemini's multimodal capabilities
    
    Args:
        frames: List of base64-encoded video frames
        audio_transcript: Optional transcript of the audio
        
    Returns:
        Analysis results
    """
    try:
        import io
        from PIL import Image
        import base64
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "Google API key not found"}
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Convert base64 images to PIL Images
        pil_images = []
        for frame in frames[:3]:  # Limit to 3 frames for efficiency
            image_bytes = base64.b64decode(frame)
            image = Image.open(io.BytesIO(image_bytes))
            pil_images.append(image)
        
        # Create the prompt
        prompt = """
        Analyze these frames from a video. Describe what's happening, identify key subjects, scenes, 
        and potential editing requirements. 
        """
        
        if audio_transcript:
            prompt += f"\n\nAUDIO TRANSCRIPT: {audio_transcript}\n"
        
        prompt += "\nReturn a JSON with 'subjects', 'scene_description', 'suggested_edits'."
        
        # Make API call with optimized parameters for speed
        try:
            model = genai.GenerativeModel('gemini-1.5-pro',
                                         generation_config={
                                             "temperature": 0.0,
                                             "max_output_tokens": 1024
                                         })
            response = model.generate_content([prompt] + pil_images)
        except Exception as model_error:
            try:
                model = genai.GenerativeModel('gemini-1.0-pro-vision',
                                            generation_config={
                                                "temperature": 0.0,
                                                "max_output_tokens": 1024
                                            })
                response = model.generate_content([prompt] + pil_images)
            except:
                raise Exception(f"No Gemini vision model available: {str(model_error)}")
        
        # Extract response
        response_text = response.text
        
        # Handle potential markdown formatting in response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            json_str = response_text
        
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            return {
                "analysis": response_text,
                "error": "Response was not in valid JSON format"
            }
    
    except Exception as e:
        return {"error": f"Gemini vision analysis error: {str(e)}"}