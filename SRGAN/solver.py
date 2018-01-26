from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
import torchvision.transforms as transforms
from SRGAN.model import Generator, Discriminator
from misc import progress_bar


class SRGANTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        self.g = None
        self.d = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.epoch_pretrain = 10
        self.criterionG = None
        self.criterionD = None
        self.optimizerG = None
        self.optimizerD = None
        self.feature_extractor = None
        self.scheduler = None
        self.GPU_IN_USE = torch.cuda.is_available()
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.num_residuals = 16
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.g = Generator(n_residual_blocks=self.num_residuals, upsample_factor=self.upscale_factor, base_filter=64)
        self.d = Discriminator(base_filter=64)
        self.feature_extractor = vgg16(pretrained=True)
        self.g.weight_init(mean=0.0, std=0.2)
        self.d.weight_init(mean=0.0, std=0.2)
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            self.g.cuda()
            self.d.cuda()
            self.feature_extractor.cuda()
            cudnn.benchmark = True
            self.criterionG.cuda()
            self.criterionD.cuda()

        self.optimizerG = optim.Adam(self.g.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizerD = optim.Adam(self.d.parameters(), lr=self.lr / 100)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=[50, 75, 100], gamma=0.5)  # lr decay
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    @staticmethod
    def to_variable(x):
        """Convert tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    @staticmethod
    def to_data(x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def save(self):
        g_model_out_path = "SRGAN_Generator_model_path.pth"
        d_model_out_path = "SRGAN_Discriminator_model_path.pth"
        torch.save(self.g, g_model_out_path)
        torch.save(self.d, d_model_out_path)
        print("Checkpoint saved to {}".format(g_model_out_path))
        print("Checkpoint saved to {}".format(d_model_out_path))

    def pretrain(self):
        self.g.train()
        for batch_num, (data, target) in enumerate(self.training_loader):
            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            self.g.zero_grad()
            loss = self.criterionG(self.g(data), target)
            loss.backward()
            self.optimizerG.step()
            print("{}/{} pretrained".format(batch_num + 1, self.epoch_pretrain))

    def train(self):
        """
        data: [torch.cuda.FloatTensor], 4 batches: [64, 64, 64, 8]
        """
        # models setup
        self.g.train()
        self.d.train()
        g_train_loss = 0
        d_train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            # setup noise
            real_label = self.to_variable(torch.ones(data.size(0)))
            fake_label = self.to_variable(torch.zeros(data.size(0)))
            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).cuda()

            # Train Discriminator
            self.optimizerD.zero_grad()
            d_real = self.d(target)
            d_real_loss = self.criterionD(d_real, real_label)

            d_fake = self.d(self.g(data))
            d_fake_loss = self.criterionD(d_fake, fake_label)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.data[0]
            d_total.backward()
            self.optimizerD.step()

            # Train generator
            self.optimizerG.zero_grad()
            g_real = self.g(data)
            g_fake = self.d(g_real)
            gan_loss = self.criterionD(g_fake, real_label)
            mse_loss = self.criterionG(g_real, target)

            recon_vgg = Variable(g_real.data.cuda())
            real_feature = self.feature_extractor(target)
            fake_feature = self.feature_extractor(recon_vgg)
            vgg_loss = self.criterionG(fake_feature, real_feature.detach())

            g_total = mse_loss + 6e-3 * vgg_loss + 1e-3 * gan_loss
            g_total += g_total.data[0]
            g_total.backward()
            self.optimizerG.step()

            progress_bar(batch_num, len(self.training_loader), 'G_Loss: %.4f | D_Loss: %.4f' % (g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1)))

        print("    Average G_Loss: {:.4f}".format(g_train_loss / len(self.training_loader)))

    def test(self):
        """
        data: [torch.cuda.FloatTensor], 10 batches: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        """
        self.g.eval()
        avg_psnr = 0
        for batch_num, (data, target) in enumerate(self.testing_loader):
            if self.GPU_IN_USE:
                data, target = Variable(data).cuda(), Variable(target).cuda()

            prediction = self.g(data)
            mse = self.criterionG(prediction, target)
            psnr = 10 * log10(1 / mse.data[0])
            avg_psnr += psnr
            progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def validate(self):
        self.build_model()
        # for epoch in range(0, self.epoch_pretrain):
        #     self.pretrain()

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save()
