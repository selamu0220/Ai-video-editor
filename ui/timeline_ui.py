"""
Advanced timeline UI components for the AI Video Editor
Styled after professional video editing software like Adobe Premiere and DaVinci Resolve
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import tempfile
import os
import base64
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import io

# Global style constants for consistent appearance
BACKGROUND_COLOR = "#1E1E1E"
TEXT_COLOR = "#FFFFFF"
TRACK_COLORS = ["#4E8EF7", "#F7754E", "#4EF775", "#AB4EF7", "#F7D54E"]
TIMELINE_HEIGHT = 600

def init_timeline_state():
    """Initialize timeline state in session state if not already present"""
    if "timeline_zoom" not in st.session_state:
        st.session_state.timeline_zoom = 1.0
    
    if "timeline_scroll" not in st.session_state:
        st.session_state.timeline_scroll = 0.0
    
    if "timeline_active_track" not in st.session_state:
        st.session_state.timeline_active_track = 0
    
    if "timeline_playhead" not in st.session_state:
        st.session_state.timeline_playhead = 0.0
    
    if "timeline_visible_tracks" not in st.session_state:
        st.session_state.timeline_visible_tracks = []
    
    if "timeline_selected_clips" not in st.session_state:
        st.session_state.timeline_selected_clips = []

def inject_custom_css():
    """Inject custom CSS for timeline styling"""
    st.markdown("""
    <style>
    /* Dark theme for professional look */
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Timeline track styling */
    .timeline-track {
        background-color: #2D2D2D;
        border-radius: 4px;
        margin-bottom: 4px;
        min-height: 40px;
    }
    
    /* Timeline clip styling */
    .timeline-clip {
        background-color: #4E8EF7;
        border-radius: 3px;
        padding: 2px 5px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        font-size: 12px;
        cursor: pointer;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Timeline clip hover effect */
    .timeline-clip:hover {
        filter: brightness(1.1);
    }
    
    /* Timeline clip selected state */
    .timeline-clip.selected {
        border: 2px solid #FFFFFF;
    }
    
    /* Playhead styling */
    .timeline-playhead {
        position: absolute;
        width: 2px;
        background-color: #FF3B30;
        height: 100%;
        top: 0;
        z-index: 100;
    }
    
    /* Timeline markers */
    .timeline-marker {
        position: absolute;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #FFCC00;
        z-index: 99;
        cursor: pointer;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2D2D2D;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555555;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #777777;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #333333;
        color: #FFFFFF;
        border: 1px solid #555555;
    }
    
    .stButton button:hover {
        background-color: #444444;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 1rem;
    }
    
    /* Container borders */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_timeline_view(timeline_data: Dict[str, Any], video_duration: float) -> None:
    """
    Generate advanced timeline visualization similar to professional video editors
    
    Args:
        timeline_data: Timeline data containing tracks, clips, markers, etc.
        video_duration: Duration of the video in seconds
    """
    init_timeline_state()
    inject_custom_css()
    
    # Timeline controls
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
        
        with col1:
            st.slider("Zoom", 0.1, 5.0, st.session_state.timeline_zoom, 0.1, 
                     key="timeline_zoom_slider", 
                     help="Zoom in/out of timeline")
        
        with col2:
            # Ensure max value is always slightly greater than min value (0.0)
            max_scroll_value = max(0.1, video_duration * st.session_state.timeline_zoom - 10)
            st.slider("Posición", 0.0, max_scroll_value, 
                     st.session_state.timeline_scroll, 0.1,
                     key="timeline_scroll_slider",
                     help="Desplazarse por la línea de tiempo")
        
        with col3:
            st.markdown("### Control de reproducción")
            play_col1, play_col2, play_col3, play_col4, play_col5 = st.columns(5)
            with play_col1:
                st.button("⏮️", key="timeline_to_start", help="Ir al inicio")
            with play_col2:
                st.button("⏪", key="timeline_rewind", help="Retroceder")
            with play_col3:
                st.button("▶️", key="timeline_play", help="Reproducir")
            with play_col4:
                st.button("⏩", key="timeline_forward", help="Avanzar")
            with play_col5:
                st.button("⏭️", key="timeline_to_end", help="Ir al final")
        
        with col4:
            st.markdown("### Tiempo actual")
            current_time = st.session_state.timeline_playhead
            formatted_time = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}.{int((current_time % 1) * 100):02d}"
            st.markdown(f"<h2 style='color: #FF3B30; text-align: center;'>{formatted_time}</h2>", unsafe_allow_html=True)
    
    # Main timeline view
    with st.container():
        # Generate time ruler
        create_time_ruler(video_duration, st.session_state.timeline_zoom, st.session_state.timeline_scroll)
        
        # Generate tracks
        track_data = timeline_data.get("tracks", [])
        render_timeline_tracks(track_data, video_duration)
        
        # Generate playhead
        render_playhead(st.session_state.timeline_playhead)
    
    # Timeline tools
    with st.container():
        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns(4)
        
        with tool_col1:
            st.button("✂️ Cortar", key="timeline_tool_cut")
        
        with tool_col2:
            st.button("🔗 Unir", key="timeline_tool_merge")
        
        with tool_col3:
            st.button("🏷️ Marcador", key="timeline_tool_marker")
        
        with tool_col4:
            st.button("🧲 Imán", key="timeline_tool_snap", 
                     help="Activar/desactivar ajuste magnético")

def create_time_ruler(duration: float, zoom: float, scroll: float) -> None:
    """
    Create time ruler for the timeline
    
    Args:
        duration: Video duration in seconds
        zoom: Current zoom level
        scroll: Current scroll position
    """
    # Calculate visible time range, ensuring it's never zero
    visible_duration = max(0.01, min(duration, 10.0 / zoom))
    start_time = scroll
    end_time = start_time + visible_duration
    
    # Determine appropriate time increments based on zoom level
    if zoom <= 0.2:
        increment = 60.0  # 1 minute
    elif zoom <= 0.5:
        increment = 30.0  # 30 seconds
    elif zoom <= 1.0:
        increment = 10.0  # 10 seconds
    elif zoom <= 2.0:
        increment = 5.0  # 5 seconds
    else:
        increment = 1.0  # 1 second
    
    # Create time markers
    times = []
    positions = []
    labels = []
    
    current_time = start_time - (start_time % increment)
    while current_time <= end_time:
        if current_time >= 0 and current_time <= duration:
            times.append(current_time)
            pos = (current_time - start_time) / visible_duration
            positions.append(pos)
            
            # Format time as MM:SS
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            labels.append(f"{minutes:02d}:{seconds:02d}")
        
        current_time += increment
    
    # Create a DataFrame for the chart
    time_data = pd.DataFrame({
        'position': positions,
        'time': times,
        'label': labels
    })
    
    # Create the chart
    ruler = alt.Chart(time_data).mark_rule(color='#555555').encode(
        x=alt.X('position:Q', axis=None, scale=alt.Scale(domain=[0, 1])),
        tooltip=['label:N']
    )
    
    # Add labels
    text = alt.Chart(time_data).mark_text(
        align='center',
        baseline='top',
        fontSize=10,
        color='#AAAAAA'
    ).encode(
        x=alt.X('position:Q', scale=alt.Scale(domain=[0, 1])),
        text='label:N'
    )
    
    # Combine the marks
    chart = (ruler + text).properties(
        height=30
    ).configure_view(
        strokeWidth=0
    )
    
    st.altair_chart(chart, use_container_width=True)

def render_timeline_tracks(tracks: List[Dict[str, Any]], duration: float) -> None:
    """
    Render the timeline tracks with clips
    
    Args:
        tracks: List of track data including clips
        duration: Video duration in seconds
    """
    # Container for all tracks
    with st.container():
        for i, track in enumerate(tracks):
            track_color = TRACK_COLORS[i % len(TRACK_COLORS)]
            track_name = track.get("name", f"Track {i+1}")
            track_type = track.get("type", "video")
            
            with st.container():
                # Track header
                col1, col2 = st.columns([2, 8])
                with col1:
                    st.markdown(f"<div style='background-color: {track_color}; padding: 8px; border-radius: 4px;'>{track_name}</div>", 
                               unsafe_allow_html=True)
                
                with col2:
                    # Render clips on track
                    clips = track.get("clips", [])
                    render_clips_on_track(clips, track_color, duration, i)

def render_clips_on_track(clips: List[Dict[str, Any]], color: str, duration: float, track_index: int) -> None:
    """
    Render clips on a timeline track
    
    Args:
        clips: List of clip data
        color: Color for the track/clips
        duration: Video duration in seconds
        track_index: Index of the track
    """
    # Calculate visible time range
    zoom = st.session_state.timeline_zoom
    scroll = st.session_state.timeline_scroll
    visible_duration = max(0.01, min(duration, 10.0 / zoom))
    start_time = scroll
    end_time = start_time + visible_duration
    
    # Create a container for the track
    track_height = 60
    
    # Prepare clip data for chart
    clip_data = []
    for clip in clips:
        clip_start = clip.get("start_time", 0)
        clip_end = clip.get("end_time", clip_start + clip.get("duration", 1))
        clip_name = clip.get("name", "Clip")
        
        # Check if clip is in visible range
        if clip_end >= start_time and clip_start <= end_time:
            # Adjust positions to visible range
            start_pos = max(0, (clip_start - start_time) / visible_duration)
            end_pos = min(1, (clip_end - start_time) / visible_duration)
            width = end_pos - start_pos
            
            if width > 0:
                clip_data.append({
                    'start': start_pos,
                    'width': width,
                    'name': clip_name,
                    'id': clip.get("id", ""),
                    'selected': clip.get("id", "") in st.session_state.timeline_selected_clips
                })
    
    # If no clips in visible range, show empty track
    if not clip_data:
        st.markdown(f"""
        <div style='height: {track_height}px; background-color: #2D2D2D; border-radius: 4px; margin-bottom: 4px;'>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Render each clip with custom HTML/CSS for better styling control
    clip_html = "<div style='position: relative; height: {}px; background-color: #2D2D2D; border-radius: 4px; margin-bottom: 4px;'>".format(track_height)
    
    for clip in clip_data:
        left_percent = clip['start'] * 100
        width_percent = clip['width'] * 100
        
        # Add selected styling if clip is selected
        selected_class = "selected" if clip['selected'] else ""
        
        clip_html += f"""
        <div class='timeline-clip {selected_class}' 
             style='position: absolute; left: {left_percent}%; width: {width_percent}%; 
                    background-color: {color}; height: 80%; top: 10%;'
             data-clip-id='{clip["id"]}'>
            {clip['name']}
        </div>
        """
    
    clip_html += "</div>"
    st.markdown(clip_html, unsafe_allow_html=True)

def render_playhead(position: float) -> None:
    """
    Render the playhead at the given position
    
    Args:
        position: Current playhead position in seconds
    """
    # Calculate playhead position based on visible range
    zoom = st.session_state.timeline_zoom
    scroll = st.session_state.timeline_scroll
    visible_duration = 10.0 / zoom
    
    # Calculate relative position (0-1)
    rel_position = (position - scroll) / visible_duration
    
    # Only show if playhead is in visible range
    if rel_position >= 0 and rel_position <= 1:
        left_percent = rel_position * 100
        
        playhead_html = f"""
        <div class='timeline-playhead' style='left: {left_percent}%;'></div>
        """
        
        st.markdown(playhead_html, unsafe_allow_html=True)

def create_transport_controls(update_playhead_callback) -> None:
    """
    Create transport controls (play, pause, etc.) for the timeline
    
    Args:
        update_playhead_callback: Callback function to update playhead position
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⏮️", key="transport_start"):
            st.session_state.timeline_playhead = 0.0
            update_playhead_callback(0.0)
    
    with col2:
        if st.button("⏪", key="transport_backward"):
            new_pos = max(0, st.session_state.timeline_playhead - 5.0)
            st.session_state.timeline_playhead = new_pos
            update_playhead_callback(new_pos)
    
    with col3:
        if "playing" not in st.session_state:
            st.session_state.playing = False
        
        if st.session_state.playing:
            if st.button("⏸️", key="transport_pause"):
                st.session_state.playing = False
        else:
            if st.button("▶️", key="transport_play"):
                st.session_state.playing = True
    
    with col4:
        if st.button("⏩", key="transport_forward"):
            new_pos = st.session_state.timeline_playhead + 5.0
            st.session_state.timeline_playhead = new_pos
            update_playhead_callback(new_pos)
    
    with col5:
        if st.button("⏭️", key="transport_end"):
            # This would need the duration passed in
            pass

def get_frame_at_position(video_path: str, position: float) -> Optional[str]:
    """
    Extract a frame from the video at the specified position
    
    Args:
        video_path: Path to the video file
        position: Position in seconds
        
    Returns:
        Base64 encoded image or None if extraction fails
    """
    try:
        import cv2
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame number
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(position * fps)
        
        # Set the position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        success, frame = cap.read()
        
        # Clean up
        cap.release()
        
        if success:
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        
        return None
    
    except Exception as e:
        st.error(f"Error extracting frame: {str(e)}")
        return None