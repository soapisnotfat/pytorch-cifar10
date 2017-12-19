import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms

import argparse

from models import *
from utils import progress_bar

# Global variables
EPOCH = 200
BATCH_SIZE = 100
ACCURACY = 0  # overall prediction accuracy
GPU_IN_USE = torch.cuda.is_available()  # whether using GPU
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ===========================================================
# parser initialization
# ===========================================================
parser = argparse.ArgumentParser(description="cifar-10 practice")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=EPOCH, type=int)
args = parser.parse_args()

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])  # dataset training transform

test_transform = transforms.Compose([transforms.ToTensor()])  # dataset testing transform


# ===========================================================
# Prepare train dataset & test dataset
# ===========================================================
print("***** prepare data ******\n")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ===========================================================
# Prepare model
# ===========================================================
print("***** prepare model *****\n")
Net = VGG('VGG11')

if GPU_IN_USE:
    Net.cuda()
    cudnn.benchmark = True

optimizer = optim.Adam(Net.parameters(), lr=args.lr)  # Adam optimization
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)  # lr decay
loss_function = nn.CrossEntropyLoss()


# Train
# ===========================================================
# data: [torch.cuda.FloatTensor of size 100x3x32x32 (GPU 0)]
# target: [torch.cuda.LongTensor of size 100 (GPU 0)]
# output: [torch.cuda.FloatTensor of size 100x10 (GPU 0)]
# prediction: [[torch.cuda.LongTensor of size 100 (GPU 0)],
#              [torch.cuda.LongTensor of size 100 (GPU 0)]]
# ===========================================================
def train():
    print("***** begin train  ******")
    Net.train()
    train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, (data, target) in enumerate(train_loader):
        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = Net(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        # TODO: introduce torch.max in blog
        prediction = torch.max(output.data, 1)  # second param "1" represent the dimension to reduce
        total += target.size(0)
        train_correct += prediction[1].eq(target.data).cpu().sum()

        progress_bar(batch_num, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

    return train_loss, train_correct / total


# test
# ===========================================================
# data: [torch.cuda.FloatTensor of size 100x3x32x32 (GPU 0)]
# target: [torch.cuda.LongTensor of size 100 (GPU 0)]
# output: [torch.cuda.FloatTensor of size 100x10 (GPU 0)]
# prediction: [[torch.cuda.LongTensor of size 100 (GPU 0)],
#              [torch.cuda.LongTensor of size 100 (GPU 0)]]
# ===========================================================
def test():
    print("\n****** begin test *******")
    Net.eval()
    test_loss = 0
    test_correct = 0
    total = 0

    for batch_num, (data, target) in enumerate(test_loader):
        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = Net(data)
        loss = loss_function(output, target)

        test_loss += loss.data[0]
        prediction = torch.max(output.data, 1)
        total += target.size(0)
        test_correct += prediction[1].eq(target.data).cpu().sum()

        progress_bar(batch_num, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_correct / (batch_num + 1), 100. * test_correct / total, test_correct, total))

    return test_loss, test_correct / total


for epoch in range(0, EPOCH):
    scheduler.step()
    print("\n\n epoch : %d/200" % (epoch + 1))
    print(train())
    print(test())
