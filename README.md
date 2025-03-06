# 絵画向けの室内レイアウト推定

本プロジェクトでは、**絵画における室内レイアウト推定**（壁・床・天井の領域分割）を行います。  
通常のレイアウト推定モデルは写真を対象に学習されており、絵画に適用すると誤検出が多発します。  
本手法では 線検出（[M-LSD](https://github.com/navervision/mlsd)）を用いて**構造線を検出**し、写真向けに学習された室内レイアウト推定モデル([NonCuboid Room Layout Estimation](https://github.com/CYang0515/NonCuboidRoom))を絵画にも適用できるように、入力画像を最適化します。


![Image](https://github.com/user-attachments/assets/12bad0fa-3c44-4bcc-8660-73ffe9f7d40a)



## インストール
このコードは、Windows11,  PyTorch v1.7, CUDA10.2, cuDNN v8.9.6でテストされている。
```
conda env create -n PaintedLayout -f PaintRoomLayout.yaml
```



## 使用するモデル
本プロジェクトでは以下の 2種類の学習済みモデル を使用します。
1. **線検出モデル ([M-LSD](https://github.com/navervision/mlsd))**

   
2. **室内レイアウト推定モデル ([NonCuboid Room Layout Estimation](https://github.com/CYang0515/NonCuboidRoom))**
  - 本論文の実験では[Structured3Dデータセットで学習されたモデル](https://drive.google.com/file/d/1DZnnOUMh6llVwhBvb-yo9ENVmN4o42x8/view "pretrained model")を使用した。



## テスト方法

### ファイル構成

指定した入力画像ディレクトリに、入力絵画と正解レイアウトを作成することで評価することができます。

```
INPUT_DIR/
├── *.jpeg  # 入力絵画
└── *.json  # 正解レイアウト
```


### コマンド構成
以下のコマンドで、**指定したモデルを用いてテストを実行**できます。

```sh
python test.py -r PATH_TO_LAYOUT_MODEL -lm PATH_TO_LINE_MODEL -i INPUT_DIR --visual
```

### オプション説明
- `-r PATH_TO_LAYOUT_MODEL` : **室内レイアウト推定モデルのパス**
- `-lm PATH_TO_LINE_MODEL` : **線検出モデルのパス**
- `-i INPUT_DIR` : **入力画像ディレクトリ**
- `--visual` : **可視化結果を保存するオプション**
