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
from utils import progress_bar


# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
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
training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)


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
# data: [torch.cuda.FloatTensor], 4 batches: [64, 64, 64, 8]
# ===========================================================
def train():
    model.train()
    train_loss = 0
    for batch_num, (data, target) in enumerate(training_data_loader):
        if GPU_IN_USE:
            data, target = Variable(data).cuda(), Variable(target).cuda()

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        progress_bar(batch_num, len(training_data_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

    print("    Average Loss: {:.4f}".format(train_loss / len(training_data_loader)))


# ===========================================================
# Test
# data: [torch.cuda.FloatTensor], 10 batches: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
# ===========================================================
def test():
    model.eval()
    avg_psnr = 0
    for batch_num, (data, target) in enumerate(testing_data_loader):
        if GPU_IN_USE:
            data, target = Variable(data).cuda(), Variable(target).cuda()

        prediction = model(data)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        progress_bar(batch_num, len(testing_data_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

    print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


# ===========================================================
# Save model
# ===========================================================
def save():
    model_out_path = "model_path.pth"
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# ===========================================================
# training and save model
# ===========================================================
for epoch in range(1, args.nEpochs + 1):
    print("\n===> Epoch {} starts:".format(epoch))
    train()
    test()
    if epoch == args.nEpochs:
        save()
