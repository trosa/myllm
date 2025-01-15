import torch
from torchtests import NeuralNetwork

model = NeuralNetwork(2,2)
model.load_state_dict(torch.load('model.pth'))