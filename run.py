import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models.heads import Heads, HRMerge
from models.hr_cfg import model_cfg
from models.hrnet import HighResolutionNet

# サンプルの画像読み込みと前処理
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image

# 特徴マップを可視化する関数
def visualize_feature_maps(feature_maps):
    num_feature_maps = feature_maps.shape[1]
    size = feature_maps.shape[2]

    # 特徴マップをグリッド形式で可視化
    fig, axs = plt.subplots(int(np.sqrt(num_feature_maps)), int(np.sqrt(num_feature_maps)), figsize=(15, 15))

    for i in range(num_feature_maps):
        ax = axs[i // int(np.sqrt(num_feature_maps)), i % int(np.sqrt(num_feature_maps))]
        ax.imshow(feature_maps[0, i].detach().cpu().numpy(), cmap='viridis')
        ax.axis('off')
    
    plt.show()

# ハイパーパラメータ
image_path = '\example\run\hallway.png'
pretrained_model_path = 'models/Structured3D_pretrained.pt'
target_layer = 'layer1'  # 可視化したいレイヤー名

# モデルの定義 (既に定義されているものを仮定)
model = HighResolutionNet(extra=model_cfg['backbone']['extra'])
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 画像をロード
image = load_image(image_path, transform)

# フォワードパスと特徴マップの取得
with torch.no_grad():
    x = model.conv1(image)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.conv2(x)
    x = model.bn2(x)
    x = model.relu(x)
    feature_maps = getattr(model, target_layer)(x)  # ここで特定のレイヤーの出力を取得

# 特徴マップを可視化
visualize_feature_maps(feature_maps)
