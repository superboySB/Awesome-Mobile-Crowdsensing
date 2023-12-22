import atexit
import threading
import os
from warp_drive.utils.device_context import make_current_context
from warp_drive.utils.context_var import my_context

# Initialize torch and CUDA context
if threading.current_thread() == threading.main_thread():
    if my_context is None:
        my_context = make_current_context()
        device = my_context.get_device()
        atexit.register(my_context.pop)
