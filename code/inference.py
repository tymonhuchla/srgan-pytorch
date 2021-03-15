import torch as T
from model import Generator, Discriminator
from PIL import Image
from torchvision import transforms
from utils import show_image

PATH = '../models/'
img_path = '../data/img_align_celeba/000001.jpg'

generator = Generator(3, 64)
discriminator = Discriminator(3, 64)

generator.load_state_dict(T.load(PATH + 'gen03.15__13.pth', map_location=T.device('cpu')))

generator.eval()


def resize(img):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.GaussianBlur(kernel_size=25),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img)
    resized = transform(img)
    return resized


while True:
    predicted = generator(resize(img_path).unsqueeze(0))
    loss = discriminator(predicted).detach().cpu().numpy()[0][0][0][0]
    if loss > 0.534:
        print(loss)
        break

show_image(predicted[0].detach().cpu(), path='image.png')

