"""
Command queue system for AI Video Editor
Allows adding commands to a queue for processing while other commands are executing
"""

import threading
import time
import queue
from typing import Dict, Any, Callable, List, Optional
import uuid

class CommandQueue:
    def __init__(self):
        """Initialize command queue system"""
        self._queue = queue.Queue()
        self._results = {}
        self._processing = False
        self._current_command = None
        self._worker_thread = None
        self._command_processor_func = None
        self._on_command_complete = None
        self._command_history = []

    def set_command_processor(self, processor_func: Callable[[str, Any], Dict[str, Any]]):
        """Set the function that will process commands"""
        self._command_processor_func = processor_func

    def set_completion_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Set the callback to be called when a command completes"""
        self._on_command_complete = callback

    def add_command(self, command: str, timeline_data: Any) -> str:
        """
        Add a command to the queue
        
        Args:
            command: The natural language command
            timeline_data: The timeline data at the time of command
            
        Returns:
            Command ID
        """
        command_id = str(uuid.uuid4())
        self._queue.put({
            "id": command_id,
            "command": command,
            "timeline_data": timeline_data,
            "status": "queued",
            "timestamp": time.time()
        })
        
        # Start worker thread if not already running
        if not self._processing:
            self._start_worker()
        
        return command_id

    def _start_worker(self):
        """Start the worker thread to process commands"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._processing = True
            self._worker_thread = threading.Thread(target=self._process_queue)
            self._worker_thread.daemon = True
            self._worker_thread.start()

    def _process_queue(self):
        """Process commands in the queue"""
        while not self._queue.empty():
            try:
                # Get next command
                command_item = self._queue.get()
                command_id = command_item["id"]
                command = command_item["command"]
                timeline_data = command_item["timeline_data"]
                
                # Update status
                command_item["status"] = "processing"
                self._current_command = command_item
                
                # Process command if processor function is set
                if self._command_processor_func:
                    try:
                        # Process the command
                        result = self._command_processor_func(command, timeline_data)
                        
                        # Store result and update status
                        command_item["result"] = result
                        command_item["status"] = "completed"
                        self._results[command_id] = result
                        
                        # Add to history
                        self._command_history.append({
                            "id": command_id,
                            "command": command,
                            "result": result,
                            "timestamp": time.time()
                        })
                        
                        # Call completion callback if set
                        if self._on_command_complete:
                            self._on_command_complete(command_id, result)
                    
                    except Exception as e:
                        # Handle error
                        command_item["status"] = "error"
                        command_item["error"] = str(e)
                        self._results[command_id] = {"error": str(e)}
                
                # Mark task as done
                self._queue.task_done()
            
            except Exception as e:
                print(f"Error in command queue worker: {str(e)}")
        
        # Reset processing state when queue is empty
        self._processing = False
        self._current_command = None

    def get_result(self, command_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific command"""
        return self._results.get(command_id)

    def get_current_command(self) -> Optional[Dict[str, Any]]:
        """Get currently processing command"""
        return self._current_command

    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the command queue"""
        return {
            "processing": self._processing,
            "queue_size": self._queue.qsize(),
            "current_command": self._current_command
        }
    
    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get history of completed commands"""
        return self._command_history
    
    def clear_history(self):
        """Clear command history"""
        self._command_history = []
    
    def is_processing(self) -> bool:
        """Check if a command is currently being processed"""
        return self._processing