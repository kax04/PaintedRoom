# 絵画向けの室内レイアウト推定

絵画に対し、室内レイアウト推定（室内二次元画像を壁・床・天井に分割する技術）をする。

![Image](https://github.com/user-attachments/assets/12bad0fa-3c44-4bcc-8660-73ffe9f7d40a)

## 準備
```
conda env create -n PaintedLayout -f BldgSeg.yaml
```

## テスト

--room_layout_model DIR : 使用する学習済みの室内レイアウトモデルを指定, 
-i DIR : 入力データのディレクトリを指定, 
--visual : 結果を可視化するオプション

```
python test.py --room_layout_model DIR -i DIR --visual
```
