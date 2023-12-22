import threading
import torch


def check_cuda_in_thread():
    from warp_drive.utils import autoinit_pycuda
    # print(f"In worker thread, CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    from warp_drive.utils import autoinit_pycuda

    # print(f"In main thread, CUDA available: {torch.cuda.is_available()}")
    # Create a worker thread to check CUDA availability
    thread = threading.Thread(target=check_cuda_in_thread)
    thread.start()
    thread.join()
