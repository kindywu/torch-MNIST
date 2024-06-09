import torch

print("torch version:",torch.__version__)
print("cuda is available:",torch.cuda.is_available())

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu") 
print("device name:",DEVICE)
