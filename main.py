import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np

import argparse

from models import *
from misc import progress_bar


# ===========================================================
# Global variables
# ===========================================================
EPOCH = 200  # number of times for each run-through
BATCH_SIZE = 100  # number of images for each epoch
ACCURACY = 0  # overall prediction accuracy
GPU_IN_USE = torch.cuda.is_available()  # whether using GPU
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # 10 classes containing in CIFAR-10 dataset


# ===========================================================
# parser initialization
# ===========================================================
parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=EPOCH, type=int, help='number of epochs tp train for')
parser.add_argument('--trainBatchSize', default=BATCH_SIZE, type=int, help='training batch size')
parser.add_argument('--testBatchSize', default=BATCH_SIZE, type=int, help='testing batch size')
args = parser.parse_args()

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])  # dataset training transform
test_transform = transforms.Compose([transforms.ToTensor()])  # dataset testing transform


# ===========================================================
# Prepare train dataset & test dataset
# ===========================================================
print("***** prepare data ******")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.trainBatchSize, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.testBatchSize, shuffle=False)
print("data preparation......Finished")

# ===========================================================
# Prepare model
# ===========================================================
if GPU_IN_USE:
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

print("\n***** prepare model *****")
# Net = LeNet().to(device)

# Net = AlexNet().to(device)

# Net = VGG11().to(device)
# Net = VGG13().to(device)
# Net = VGG16().to(device)
# Net = VGG19().to(device)

# Net = GoogLeNet().to(device)

# Net = resnet18().to(device)
# Net = resnet34().to(device)
# Net = resnet50().to(device)
# Net = resnet101().to(device)
# Net = resnet152().to(device)

# Net = DenseNet121().to(device)
# Net = DenseNet161().to(device)
# Net = DenseNet169().to(device)
# Net = DenseNet201().to(device)

Net = WideResNet(depth=28, num_classes=10).to(device)

optimizer = optim.Adam(Net.parameters(), lr=args.lr)  # Adam optimization
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)  # lr decay
loss_function = nn.CrossEntropyLoss()
print("model preparation......Finished")


# Train
# ===========================================================
# data: [torch.cuda.FloatTensor of size 100x3x32x32 (GPU 0)]
# target: [torch.cuda.LongTensor of size 100 (GPU 0)]
# output: [torch.cuda.FloatTensor of size 100x10 (GPU 0)]
# prediction: [[torch.cuda.LongTensor of size 100 (GPU 0)],
#              [torch.cuda.LongTensor of size 100 (GPU 0)]]
# ===========================================================
def train():
    print("train:")
    Net.train()
    train_loss = 0
    train_correct = 0
    total = 0

    for batch_num, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = Net(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
        total += target.size(0)

        # train_correct incremented by one if predicted right
        train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

        progress_bar(batch_num, len(train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
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
    print("test:")
    Net.eval()
    test_loss = 0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = Net(data)
            loss = loss_function(output, target)
            test_loss += loss.item()
            prediction = torch.max(output, 1)
            total += target.size(0)
            test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

    return test_loss, test_correct / total


# ===========================================================
# Save model
# ===========================================================
def save():
    model_out_path = "model.pth"
    torch.save(Net, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# ===========================================================
# training and save model
# ===========================================================
for epoch in range(1, args.epoch + 1):
    scheduler.step(epoch)
    print("\n===> epoch: %d/200" % epoch)
    train_result = train()
    print(train_result)
    test_result = test()
    ACCURACY = max(ACCURACY, test_result[1])
    if epoch == args.epoch:
        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (ACCURACY * 100))
        save()
