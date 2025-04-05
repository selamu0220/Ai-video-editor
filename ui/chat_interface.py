"""
Chat-like interface component for the AI Video Editor
Displays video edits in a ChatGPT-like interface with navigation between edits
"""

import streamlit as st
import os
import time
from typing import Dict, Any, List, Optional, Tuple

def init_chat_state():
    """Initialize chat state in session state if not already present"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_chat_index" not in st.session_state:
        st.session_state.current_chat_index = -1

def add_chat_message(role: str, content: str, video_path: Optional[str] = None, 
                     operations: Optional[List[Dict[str, Any]]] = None,
                     thumbnail_path: Optional[str] = None):
    """
    Add a message to the chat history
    
    Args:
        role: Either "user" or "assistant"
        content: Text content of the message
        video_path: Path to the video file (for assistant messages)
        operations: List of operations performed (for assistant messages)
        thumbnail_path: Path to the thumbnail image
    """
    if "chat_history" not in st.session_state:
        init_chat_state()
    
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time()
    }
    
    if role == "assistant" and video_path:
        message["video_path"] = video_path
    
    if operations:
        message["operations"] = operations
    
    if thumbnail_path:
        message["thumbnail"] = thumbnail_path
    
    st.session_state.chat_history.append(message)
    st.session_state.current_chat_index = len(st.session_state.chat_history) - 1

def get_current_chat_message() -> Optional[Dict[str, Any]]:
    """Get the current chat message being displayed"""
    if "chat_history" not in st.session_state or "current_chat_index" not in st.session_state:
        return None
    
    if st.session_state.current_chat_index < 0 or st.session_state.current_chat_index >= len(st.session_state.chat_history):
        return None
    
    return st.session_state.chat_history[st.session_state.current_chat_index]

def navigate_chat(direction: str):
    """
    Navigate chat history in the specified direction
    
    Args:
        direction: Either "prev" or "next"
    """
    if "chat_history" not in st.session_state or "current_chat_index" not in st.session_state:
        return
    
    if direction == "prev" and st.session_state.current_chat_index > 0:
        st.session_state.current_chat_index -= 1
    elif direction == "next" and st.session_state.current_chat_index < len(st.session_state.chat_history) - 1:
        st.session_state.current_chat_index += 1

def render_chat_interface():
    """Render the chat-like interface for video editing interactions"""
    if "chat_history" not in st.session_state:
        init_chat_state()
    
    # Create columns for left sidebar, chat area, and right sidebar
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col1:
        # Navigation buttons (left sidebar)
        if st.session_state.current_chat_index > 0:
            if st.button("⬅️ Anterior", key="prev_chat"):
                navigate_chat("prev")
    
    with col3:
        # Navigation buttons (right sidebar)
        if st.session_state.current_chat_index < len(st.session_state.chat_history) - 1:
            if st.button("Siguiente ➡️", key="next_chat"):
                navigate_chat("next")
    
    with col2:
        # Display the current chat message
        current_msg = get_current_chat_message()
        if current_msg:
            with st.container(border=True):
                # Display message header
                if current_msg["role"] == "user":
                    st.markdown("### 👤 Usuario")
                else:
                    st.markdown("### 🤖 AI Video Editor")
                
                # Display message content
                st.write(current_msg["content"])
                
                # If assistant message with video, display video and operations
                if current_msg["role"] == "assistant" and "video_path" in current_msg:
                    video_path = current_msg["video_path"]
                    if os.path.exists(video_path):
                        # DEBUG: Print the video path being used for st.video
                        st.write(f"DEBUG: Rendering video from chat message: {video_path}")
                        
                        # Video display with fixed height for consistent UI
                        st.video(video_path, start_time=0)
                    else:
                        st.warning(f"Video no encontrado: {video_path}")
                
                # If thumbnail exists, display it
                if "thumbnail" in current_msg and os.path.exists(current_msg["thumbnail"]):
                    with st.expander("Ver miniatura"):
                        st.image(current_msg["thumbnail"])
                
                # If operations exist, display them
                if "operations" in current_msg:
                    with st.expander("Ver operaciones realizadas"):
                        for i, op in enumerate(current_msg["operations"]):
                            st.write(f"**Operación {i+1}:** {op.get('type', 'Desconocida')}")
                            st.json(op)
        
        # Navigation indicators
        if st.session_state.chat_history:
            col_indicators = st.columns(len(st.session_state.chat_history))
            for i in range(len(st.session_state.chat_history)):
                with col_indicators[i]:
                    if i == st.session_state.current_chat_index:
                        st.markdown("🔵")
                    else:
                        st.markdown("⚪")

def create_chat_input() -> Tuple[bool, str]:
    """
    Create the chat input area
    
    Returns:
        Tuple of (submitted, command)
    """
    with st.container(border=False):
        command = st.text_area("Escribe un comando para editar el video", 
                              placeholder="Ej: 'Corta todas las partes silenciosas del video'",
                              key="chat_input")
        
        submitted = st.button("Enviar comando", key="submit_command")
        
        return submitted, command