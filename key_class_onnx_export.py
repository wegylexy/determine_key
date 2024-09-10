import torch
from key_class_neural_network import key_class_nn
c = torch.load('key_class_nn.pth')
cn = key_class_nn()
cn.load_state_dict(c['state_dict'])
cn.eval()
cd = torch.zeros(1, 3, 256, 862)
torch.onnx.export(cn, cd, 'key_class.onnx', export_params=True)