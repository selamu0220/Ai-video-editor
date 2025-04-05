import json
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import shutil

class Timeline:
    def __init__(self):
        """Initialize a new timeline"""
        self.segments = []
        self.markers = []
        self.effects = []
        self.duration = 0
        self.total_frames = 0
        self.fps = 0
    
    def add_segment(self, segment: Dict[str, Any]) -> None:
        """Add a segment to the timeline"""
        self.segments.append(segment)
        
        # Update duration if needed
        segment_end = segment.get("end_time", 0)
        if segment_end > self.duration:
            self.duration = segment_end
    
    def add_marker(self, time: float, label: str) -> None:
        """Add a marker at a specific time"""
        self.markers.append({
            "time": time,
            "label": label
        })
    
    def add_effect(self, effect: Dict[str, Any]) -> None:
        """Add an effect to the timeline"""
        self.effects.append(effect)
    
    def get_serializable_data(self) -> Dict[str, Any]:
        """Get timeline data in a serializable format"""
        return {
            "segments": self.segments,
            "markers": self.markers,
            "effects": self.effects,
            "duration": self.duration,
            "total_frames": self.total_frames,
            "fps": self.fps
        }

def initialize_timeline_data(video_path: str) -> Timeline:
    """
    Initialize timeline data from a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Timeline object with initialized data
    """
    # Create a new timeline
    timeline = Timeline()
    
    # Open the video to get its properties
    # clip = VideoFileClip(video_path)
    
    # Set basic timeline properties
    timeline.duration = 0  # clip.duration
    timeline.fps = 0  # clip.fps
    timeline.total_frames = 0  # int(clip.duration * clip.fps)
    
    # Add the entire video as a single segment
    timeline.add_segment({
        "id": 1,
        "start_time": 0,
        "end_time": timeline.duration,
        "type": "video",
        "source": video_path,
        "speed": 1.0
    })
    
    # Close the clip
    # clip.close()
    
    return timeline

def update_timeline_data(timeline: Timeline, update_data: Dict[str, Any]) -> Timeline:
    """
    Update timeline data with new information
    
    Args:
        timeline: Existing timeline object
        update_data: New data to update
        
    Returns:
        Updated timeline object
    """
    # Process operations
    operations = update_data.get("operations", [])
    
    for operation in operations:
        op_type = operation.get("type", "")
        params = operation.get("params", {})
        timeline_position = operation.get("timeline_position", "all")
        
        # Handle timeline-specific operations
        if op_type == "add_marker":
            time = params.get("time", 0)
            label = params.get("label", "Marker")
            timeline.add_marker(time, label)
        
        elif op_type == "add_effect":
            effect_type = params.get("effect_type", "")
            start_time = params.get("start_time", 0)
            end_time = params.get("end_time", timeline.duration)
            effect_params = params.get("effect_params", {})
            
            timeline.add_effect({
                "type": effect_type,
                "start_time": start_time,
                "end_time": end_time,
                "params": effect_params
            })
        
        elif op_type == "trim":
            # If we're trimming the video, update the segments
            start_time = params.get("start_time", 0)
            end_time = params.get("end_time", timeline.duration)
            
            # Replace the existing segments with a new trimmed segment
            timeline.segments = [{
                "id": 1,
                "start_time": start_time,
                "end_time": end_time,
                "type": "video",
                "source": timeline.segments[0].get("source", ""),
                "speed": timeline.segments[0].get("speed", 1.0)
            }]
            
            # Update the duration
            timeline.duration = end_time - start_time
    
    return timeline

def render_timeline(timeline: Timeline) -> None:
    """
    Render the timeline visualization in Streamlit
    
    Args:
        timeline: Timeline object to render
    """
    # Calculate display parameters
    timeline_width = 800
    timeline_height = 100
    pixels_per_second = timeline_width / max(1, timeline.duration)
    
    # Create a container for the timeline
    timeline_container = st.container()
    
    with timeline_container:
        # Draw the main timeline bar
        st.markdown(f"""
        <div style="
            width: {timeline_width}px;
            height: {timeline_height}px;
            background-color: #f0f0f0;
            position: relative;
            border-radius: 5px;
            margin-bottom: 20px;
        ">
        """, unsafe_allow_html=True)
        
        # Draw segments
        for segment in timeline.segments:
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", timeline.duration)
            segment_type = segment.get("type", "video")
            
            # Calculate position and width
            left_pos = start_time * pixels_per_second
            width = (end_time - start_time) * pixels_per_second
            
            # Choose color based on type
            color = "#4CAF50" if segment_type == "video" else "#2196F3"
            
            st.markdown(f"""
            <div style="
                position: absolute;
                left: {left_pos}px;
                top: 20px;
                width: {width}px;
                height: 60px;
                background-color: {color};
                border-radius: 3px;
            "></div>
            """, unsafe_allow_html=True)
        
        # Draw markers
        for marker in timeline.markers:
            time = marker.get("time", 0)
            label = marker.get("label", "Marker")
            
            # Calculate position
            left_pos = time * pixels_per_second
            
            st.markdown(f"""
            <div style="
                position: absolute;
                left: {left_pos}px;
                top: 0;
                width: 2px;
                height: {timeline_height}px;
                background-color: #FF5722;
            "></div>
            <div style="
                position: absolute;
                left: {left_pos - 50}px;
                top: {timeline_height + 5}px;
                width: 100px;
                text-align: center;
                font-size: 12px;
                color: #FF5722;
            ">{label}</div>
            """, unsafe_allow_html=True)
        
        # Draw effects
        for effect in timeline.effects:
            effect_type = effect.get("type", "")
            start_time = effect.get("start_time", 0)
            end_time = effect.get("end_time", timeline.duration)
            
            # Calculate position and width
            left_pos = start_time * pixels_per_second
            width = (end_time - start_time) * pixels_per_second
            
            st.markdown(f"""
            <div style="
                position: absolute;
                left: {left_pos}px;
                top: 85px;
                width: {width}px;
                height: 10px;
                background-color: #9C27B0;
                border-radius: 2px;
            "></div>
            <div style="
                position: absolute;
                left: {left_pos}px;
                top: 97px;
                width: {width}px;
                text-align: center;
                font-size: 10px;
                color: #9C27B0;
                overflow: hidden;
                white-space: nowrap;
                text-overflow: ellipsis;
            ">{effect_type}</div>
            """, unsafe_allow_html=True)
        
        # Close the container div
        st.markdown("""</div>""", unsafe_allow_html=True)
        
        # Display time markers
        time_markers = []
        num_markers = 10
        for i in range(num_markers + 1):
            time = (i / num_markers) * timeline.duration
            time_markers.append(f"{time:.1f}s")
        
        # Display the time markers
        st.markdown(f"""
        <div style="
            width: {timeline_width}px;
            display: flex;
            justify-content: space-between;
            margin-top: -15px;
            font-size: 12px;
            color: #666;
        ">
            {"".join(f'<div>{marker}</div>' for marker in time_markers)}
        </div>
        """, unsafe_allow_html=True)
        
        # Display timeline information
        st.markdown(f"""
        <div style="margin-top: 20px; font-size: 14px;">
            <strong>Duration:</strong> {timeline.duration:.2f} seconds | 
            <strong>FPS:</strong> {timeline.fps:.2f} | 
            <strong>Total Frames:</strong> {timeline.total_frames}
        </div>
        """, unsafe_allow_html=True)

def add_timeline_marker(timeline: Timeline, time: float, label: str) -> Timeline:
    """
    Add a marker to the timeline
    
    Args:
        timeline: Timeline object
        time: Time position for the marker (in seconds)
        label: Label for the marker
        
    Returns:
        Updated timeline object
    """
    timeline.add_marker(time, label)
    return timeline

def add_timeline_effect(timeline: Timeline, effect_type: str, start_time: float, 
                      end_time: float, params: Dict[str, Any]) -> Timeline:
    """
    Add an effect to the timeline
    
    Args:
        timeline: Timeline object
        effect_type: Type of effect
        start_time: Start time for the effect (in seconds)
        end_time: End time for the effect (in seconds)
        params: Effect parameters
        
    Returns:
        Updated timeline object
    """
    timeline.add_effect({
        "type": effect_type,
        "start_time": start_time,
        "end_time": end_time,
        "params": params
    })
    return timeline

def split_segment(timeline: Timeline, time: float) -> Timeline:
    """
    Split a segment at the specified time
    
    Args:
        timeline: Timeline object
        time: Time to split at (in seconds)
        
    Returns:
        Updated timeline object
    """
    # Find the segment that contains the split time
    segment_to_split = None
    for segment in timeline.segments:
        start_time = segment.get("start_time", 0)
        end_time = segment.get("end_time", timeline.duration)
        
        if start_time <= time < end_time:
            segment_to_split = segment
            break
    
    if segment_to_split:
        # Create two new segments
        segment1 = segment_to_split.copy()
        segment2 = segment_to_split.copy()
        
        # Update the times
        segment1["end_time"] = time
        segment2["start_time"] = time
        
        # Generate a new ID for the second segment
        max_id = max([s.get("id", 0) for s in timeline.segments])
        segment2["id"] = max_id + 1
        
        # Remove the original segment and add the new ones
        timeline.segments.remove(segment_to_split)
        timeline.segments.append(segment1)
        timeline.segments.append(segment2)
    
    return timeline

def get_segment_at_time(timeline: Timeline, time: float) -> Optional[Dict[str, Any]]:
    """
    Get the segment at the specified time
    
    Args:
        timeline: Timeline object
        time: Time position (in seconds)
        
    Returns:
        Segment dict if found, None otherwise
    """
    for segment in timeline.segments:
        start_time = segment.get("start_time", 0)
        end_time = segment.get("end_time", timeline.duration)
        
        if start_time <= time < end_time:
            return segment
    
    return None

def export_timeline_data(timeline: Timeline) -> str:
    """
    Export timeline data as JSON
    
    Args:
        timeline: Timeline object
        
    Returns:
        JSON string representation of the timeline
    """
    return json.dumps(timeline.get_serializable_data(), indent=2)

def import_timeline_data(json_data: str) -> Timeline:
    """
    Import timeline data from JSON
    
    Args:
        json_data: JSON string representation of the timeline
        
    Returns:
        Timeline object
    """
    data = json.loads(json_data)
    
    timeline = Timeline()
    timeline.duration = data.get("duration", 0)
    timeline.total_frames = data.get("total_frames", 0)
    timeline.fps = data.get("fps", 0)
    
    # Add segments
    for segment in data.get("segments", []):
        timeline.add_segment(segment)
    
    # Add markers
    for marker in data.get("markers", []):
        timeline.add_marker(marker.get("time", 0), marker.get("label", ""))
    
    # Add effects
    for effect in data.get("effects", []):
        timeline.add_effect(effect)
    
    return timeline
