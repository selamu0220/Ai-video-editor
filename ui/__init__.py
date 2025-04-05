"""
UI components for AI Video Editor
"""

from ui.chat_interface import (
    init_chat_state, 
    add_chat_message, 
    get_current_chat_message,
    navigate_chat,
    render_chat_interface,
    create_chat_input
)

from ui.timeline_ui import (
    init_timeline_state,
    inject_custom_css,
    generate_timeline_view,
    create_time_ruler,
    render_timeline_tracks,
    render_clips_on_track,
    render_playhead,
    create_transport_controls,
    get_frame_at_position
)