# 必要なライブラリをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ファッションアイテムの日本語と英語のラベルを定義
classes_ja = ["Tシャツ/トップ", "ズボン", "プルオーバー", "ドレス", "コート",
              "サンダル", "ワイシャツ", "スニーカー", "バッグ", "アンクルブーツ"]
classes_en = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# クラス数（ラベルの数）と画像のサイズを定義
num_classes = len(classes_ja)
img_size = 28

# 画像認識のためのCNN（畳み込みニューラルネットワーク）モデルを定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 畳み込み層とバッチノーマリゼーション層を定義
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        # プーリング層（画像を縮小する層）と活性化関数を定義
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # 全結合層（画像の特徴を使って分類を行う層）とドロップアウト層（過学習を防ぐための層）を定義
        self.fc1 = nn.Linear(64*4*4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)

    # フォワードパス（入力から出力までの計算手順）を定義
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)

        # 全結合層への入力のために、形を1次元に変更
        x = x.view(-1, 64*4*4)

        # 全結合層を使用して分類を行い、結果を返す
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# モデルのインスタンスを作成
net = Net()

# 保存されたモデルの重みを読み込む
net.load_state_dict(torch.load("model_cnn.pth", map_location=torch.device("cpu")))

def predict(input_img):
    """
    画像を入力として受け取り、CNNモデルを使用してアイテムの分類を行う関数。
    """
    # 画像をモノクロに変換し、28x28のサイズにリサイズ
    img = input_img.convert("L")
    img = img.resize((img_size, img_size))

    # 画像の正規化を行うための前処理を定義
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0), (1.0))
    ])
    img = transform(img)
    img_tensor = img.reshape(1, 1, img_size, img_size)

    # モデルを評価モードにして、画像を使用して分類を行う
    net.eval()
    predicted_output = net(img_tensor)

    # 分類の結果を確率に変換し、降順にソート
    probabilities = torch.nn.functional.softmax(torch.squeeze(predicted_output))
    sorted_prob, sorted_indices = torch.sort(probabilities, descending=True)
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
