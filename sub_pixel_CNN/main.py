from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set


# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
args = parser.parse_args()
print(args)

GPU_IN_USE = torch.cuda.is_available()

torch.manual_seed(args.seed)
if GPU_IN_USE:
    torch.cuda.manual_seed(args.seed)


# ===========================================================
# Set train dataset & test dataset
# ===========================================================
print('===> Loading datasets')
train_set = get_training_set(args.upscale_factor)
test_set = get_test_set(args.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)


# ===========================================================
# Setup model
# ===========================================================
print('===> Building model')
model = Net(upscale_factor=args.upscale_factor)
criterion = nn.MSELoss()

if GPU_IN_USE:
    model.cuda()
    cudnn.benchmark = True

optimizer = optim.Adam(model.parameters(), lr=args.lr)


# ===========================================================
# Train
# ===========================================================
def train(epoch):
    epoch_loss = 0
    for batch_num, (data, target) in enumerate(training_data_loader):
        if GPU_IN_USE:
            data, target = Variable(data).cuda(), Variable(target).cuda()

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, batch_num, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


# ===========================================================
# Test
# ===========================================================
def test():
    avg_psnr = 0
    for batch_num, (data, target) in enumerate(testing_data_loader):
        if GPU_IN_USE:
            data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()

        prediction = model(data)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


# ===========================================================
# Save model
# ===========================================================
def save():
    model_out_path = "model_path.pth"
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, args.nEpochs + 1):
    train(epoch)
    test()
    save()
