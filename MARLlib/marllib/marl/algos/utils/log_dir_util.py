# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import shutil
import os

sub_dir_name = "marllib_results"

path_to_log = os.path.join("/workspace", "saved_data")
__total, __used, __free = shutil.disk_usage(path_to_log)

available_local_dir = os.path.join(path_to_log, sub_dir_name) \
    if __used / __total <= 0.95 else os.path.join("/mnt", sub_dir_name)

path_to_temp = os.path.join("/workspace", "saved_data", "tmp", "ray")
__total, __used, __free = shutil.disk_usage(path_to_temp)
path_to_temp = path_to_temp if __used / __total <= 0.95 else os.path.join("/tmp", "ray")
