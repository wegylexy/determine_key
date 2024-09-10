import torch
from key_quality_neural_network import key_quality_nn
q = torch.load('key_quality_nn.pth')
qn = key_quality_nn()
qn.load_state_dict(q['state_dict'])
qn.eval()
cd = torch.zeros(1, 3, 256, 862)
torch.onnx.export(qn, cd, 'key_quality.onnx', export_params=True)