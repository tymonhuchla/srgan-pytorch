from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch as T
import torch.optim as optim
from model import Generator, Discriminator
from loss_fn import GeneratorLoss, TVLoss
from utils import show_progress, save
import datetime
import gc
import os


class ConcatDataset(T.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


device = 'cuda' if T.cuda.is_available() else 'cpu'

BATCH_SIZE = 16
SIZE_HR = 256
SIZE_LR = 64
num_workers = 2
rootpath = '../data'
transform_hr = transforms.Compose([
                                transforms.Resize((SIZE_HR, SIZE_HR)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

data_hr = ImageFolder(rootpath, transform=transform_hr)

transform_lr = transforms.Compose([
                                  transforms.Resize((SIZE_LR, SIZE_LR)),
                                  transforms.ToTensor(),
                                  transforms.GaussianBlur(kernel_size=25),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ])

data_lr = ImageFolder(rootpath, transform=transform_lr)

full_data = ConcatDataset(data_lr, data_hr)
loader = DataLoader(full_data, BATCH_SIZE, num_workers=num_workers)

generator = Generator(3, 64).to(device)
discriminator = Discriminator(3, 64).to(device)

lr = 1e-1000

gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

generator_criterion = GeneratorLoss().to(device)

g_losses = []
d_losses = []


EPOCHS = 1000
if 'models' not in os.listdir():
    os.mkdir('models')

save_path = './models/'

# <----- TRAINING LOOP ----->

for epoch in range(1, EPOCHS):
    generator.train()
    discriminator.train()
    print(f'EPOCH [{epoch}/{EPOCHS}]')
    sum_d_loss = 0
    sum_g_loss = 0
    gc.collect()
    T.cuda.empty_cache()
    start = datetime.datetime.now()
    for idx, (item, target) in enumerate(loader):
        item = item[0].to(device)
        target = target[0].to(device)
        fake_image = generator(item)
        discriminator.zero_grad()

        real_out = discriminator(target).mean()
        fake_out = discriminator(fake_image).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)

        generator.zero_grad()
        g_loss = generator_criterion(fake_out, fake_image, target)
        g_loss.backward()

        fake_img = generator(item)
        fake_out = discriminator(fake_img).mean()
        if idx % 100 == 0:
            print(
                f'Batch {idx}/{loader.__len__()} \nLoss (Generator) {g_loss.detach().cpu()}\nLoss (Discriminator) {d_loss.detach().cpu()}'
            )
            pred = fake_img[0].detach().cpu()
            save(generator, discriminator, save_path)
            show_progress([item.detach().cpu()[0], pred, target.detach().cpu()[0]], save=True, show=False)

        gen_optimizer.step()

        sum_d_loss += d_loss.detach().cpu()
        sum_g_loss += g_loss.detach().cpu()
    print(f'Time per epoch = {start - datetime.datetime.now()}')
    g_losses.append(sum_g_loss / loader.__len__())
    d_losses.append(sum_d_loss / loader.__len__())
    print(f'D_loss {sum_d_loss}')
    print(f'G_loss {sum_g_loss}')