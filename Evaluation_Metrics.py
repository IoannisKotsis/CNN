import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

inputs=torch.tensor([[1.0, 1.0],
                    [1.2, 0.9],
                    [0.8, 1.1],
                    [3.0, 3.0],
                    [3.2, 2.8],
                    [2.9, 3.1],
                    [6.0, 0.5],
                    [6.1, 0.7],
                    [5.9, 0.6],
                    [4.0, 4.0],
                    [4.1, 4.2],
                    [3.9, 4.1]], dtype=torch.float)

labels=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1], dtype=torch.long)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(2,6)
        self.layer2=nn.Linear(6,3)

    def forward(self, x):
        out1=F.relu(self.layer1(x))
        out2=self.layer2(out1)
        return out2

diktyo=Net()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(diktyo.parameters(), lr=0.05)

diktyo.train()
for epoch in range(100):          #να προσθέσω loop για batches (προσθήκη ΜΝΙST)
    optimizer.zero_grad()
    out2=diktyo(inputs)
    loss=criterion(out2,labels)
    loss.backward()
    optimizer.step()
    if (epoch+1)%20==0:
        print(f"epoch {epoch+1} loss is: {loss.item():.5f}")

diktyo.eval()

from torchmetrics.functional import accuracy, recall, precision, f1_score

with torch.no_grad():
    out2=diktyo(inputs)
    out2_preds=out2.argmax(1)

    acc=accuracy(out2_preds,labels, task="multiclass", num_classes=3, average="micro")
    rec=recall(out2_preds,labels, task="multiclass", num_classes=3, average="micro")
    prec=precision(out2_preds,labels, task="multiclass", num_classes=3, average="micro")
    F1_Score=f1_score(out2_preds,labels, task="multiclass", num_classes=3, average="micro")
    print(f"Accuracy is: {acc:.5f} \nRecall is: {rec:.5f} \nPrecision is: {prec:.5f} \nF1 Score is: {F1_Score:.5f}")





