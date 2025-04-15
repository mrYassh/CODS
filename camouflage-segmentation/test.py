import torch
print(torch.cuda.is_available())
print(torch.version.cuda if torch.cuda.is_available() else "CUDA not available")