import numpy as np

import os

folder_path = './solution'

files = os.listdir(folder_path)

for file in files:

    new_name = file.zfill(7)

    old_file = os.path.join(folder_path, file)
    new_file = os.path.join(folder_path, new_name)

    os.rename(old_file, new_file)
