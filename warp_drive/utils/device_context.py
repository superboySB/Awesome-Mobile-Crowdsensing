import os
import subprocess

import torch
import pycuda.driver as cuda_driver

def make_current_context(device_id=None):
    # Note, 77 configuration, change when deployed to other machines
    # command = "nvidia-smi --list-gpus"
    # # Run the command and capture the output
    # try:
    #     output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
    #     # Split the output into lines
    #     lines = output.strip().split('\n')
    #     # Count the number of CUDA devices
    #     num_cuda_devices = len(lines)
    #     print(f"Number of CUDA devices: {num_cuda_devices}")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error running command, please check CUDA devices availability: {e}")
    #     raise e
    #
    # visible_str = ','.join([str(i) for i in range(num_cuda_devices)])
    # os.environ["CUDA_VISIBLE_DEVICES"] = visible_str
    torch.cuda.init()
    cuda_driver.init()
    if device_id is None:
        context = _get_primary_context_for_current_device()
    else:
        context = cuda_driver.Device(device_id).retain_primary_context()
    context.push()
    return context


def _get_primary_context_for_current_device():
    ndevices = cuda_driver.Device.count()
    if ndevices == 0:
        raise RuntimeError("No CUDA enabled device found. "
                           "Please check your installation.")

    # Is CUDA_DEVICE set?
    import os
    devn = os.environ.get("CUDA_DEVICE")

    # Is $HOME/.cuda_device set ?
    if devn is None:
        try:
            homedir = os.environ.get("HOME")
            assert homedir is not None
            devn = (open(os.path.join(homedir, ".cuda_device"))
                    .read().strip())
        except:
            pass

    # If either CUDA_DEVICE or $HOME/.cuda_device is set, try to use it ;-)
    if devn is not None:
        try:
            devn = int(devn)
        except TypeError:
            raise TypeError("CUDA device number (CUDA_DEVICE or ~/.cuda_device)"
                            " must be an integer")

        dev = cuda_driver.Device(devn)
        return dev.retain_primary_context()

    # Otherwise, try to use any available device
    else:
        for devn in range(ndevices):
            dev = cuda_driver.Device(devn)
            try:
                return dev.retain_primary_context()
            except cuda_driver.Error:
                pass

        raise RuntimeError("_get_primary_context_for_current_device() wasn't able to create a context "
                           "on any of the %d detected devices" % ndevices)
