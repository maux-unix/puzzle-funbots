import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
