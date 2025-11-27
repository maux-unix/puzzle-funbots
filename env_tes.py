import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
