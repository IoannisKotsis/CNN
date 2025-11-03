import torch
import torch.nn as nn
import torch.nn.functional as F

inputs=torch.randn(10,4, dtype=torch.float)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(4,8)
        self.layer2=nn.Linear(8,3)
        self.layer3=nn.Linear(3,1)

    def forward(self,x):
        out1=F.relu(self.layer1(x))
        out2=torch.tanh(self.layer2(out1))
        out3=torch.sigmoid(self.layer3(out2))
        return out3


model=Network()
out3=model(inputs)

print("Output shape is:", out3.shape)
print("\n Output values are:", out3)