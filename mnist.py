import torch
import torch.nn.functional as  F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets,transforms
from torchsummary import summary
import myNet

batch_size = 128
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose(
                            [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
test_data =  datasets.MNIST('./data', train=False, transform=transforms.Compose(
                            [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))


train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=True)


model = myNet.CNN().to(device)
summary(model,(1,28,28))
optimizer = optim.Adam(model.parameters())

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if (batch_idx)%30 == 0:
            print('train {} epoch : {}/{} \t loss : {:.6f}'.format(
                                                    epoch,batch_idx*len(data),len(train_loader.dataset),loss.item()))


def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():#如果.requires_grad=True但是你又不希望进行autograd的计算， 那么可以将变量包裹在 with torch.no_grad()中
        for data,target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output,target,reduction='sum').item()
            pred = output.max(1,keepdim = True)[1]
            correct +=pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nvalid loss : {:.4f} \t accuracy : {:.3f}%\n'.format(
                                                    test_loss,100. * correct / len(test_loader.dataset)))

for epoch in range(epochs):
    train(model,device,train_loader,optimizer,epoch)
    test(model,device,test_loader)




















