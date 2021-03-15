import matplotlib.pyplot as plt
import datetime
import os
import torch as T


def show_image(image):
    result = image.permute(1, 2, 0)
    plt.imshow(result, interpolation='nearest')
    plt.show()


def show_progress(imgs, path='generated_data', save=False, show=True):
    # Low res -> Predicted -> High res
    time = datetime.datetime.now().strftime("%m.%d__%H:%M")
    if show:
        fig, ax = plt.subplots(1, 3, figsize=(20,20))
        for idx, img in enumerate(imgs):
          img = img.permute(1,2,0)
          ax[idx].imshow(img, interpolation='nearest')
          ax[idx].axis('off')
        plt.show()
    if save:
        if path not in os.listdir():
          os.mkdir(path)
        fig, ax = plt.subplots(1, 3, figsize=(20,20))
        for idx, img in enumerate(imgs):
          img = img.permute(1,2,0)
          ax[idx].imshow(img, interpolation='nearest')
          ax[idx].axis('off')
        plt.savefig(path + '/' + f'img{time}.png')


def save(gen, disc, path):
    time = datetime.datetime.now().strftime("%m.%d__%H")
    T.save(gen.state_dict(), path + f'gen{time}.pth')
    T.save(disc.state_dict(), path + f'disc{time}.pth')