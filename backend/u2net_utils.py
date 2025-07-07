import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import urllib.request  # Для загрузки весов из интернета

from model.u2net import U2NETP  # Импорт самой модели

def download_model():
    model_dir = "./models"
    model_path = os.path.join(model_dir, "u2netp.pth")
    
    if not os.path.exists(model_path):
        print("Скачиваем модель U2NET...")
        os.makedirs(model_dir, exist_ok=True)
        url = "https://huggingface.co/netradrishti/u2net-saliency/resolve/7be90880f2c00a8864982011315a7b8532e16dbe/models/u2netp.pth"
        urllib.request.urlretrieve(url, model_path)
        print("Модель успешно загружена.")
    else:
        print("Модель уже загружена.")
    return model_path

def load_model():
    model_path = download_model()
    print("Загружаем модель...")
    net = U2NETP(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def remove_background(net, image: Image.Image) -> Image.Image:
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Приводим к RGB
    image_rgb = image.convert("RGB")

    orig_size = image.size
    image_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        d1, *_ = net(image_tensor)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred_np = pred.squeeze().cpu().numpy()

    mask = Image.fromarray((pred_np * 255).astype(np.uint8)).resize(orig_size, Image.BILINEAR)
    image = image.convert("RGBA")
    np_img = np.array(image)
    np_img[:, :, 3] = np.array(mask)
    print(image.mode)
    print(np_img.shape)
    return Image.fromarray(np_img)
