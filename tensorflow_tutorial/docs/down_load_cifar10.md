## CIFAR-10のダウンロード
```
cd tensor_flow_tutorial
uv run python down_load_cifar10.py
```
実行すると，
``tensor_flow_tutorial/tensorflow_datasets/cifar10/3.0.2/`` に以下が揃う．

- cifar10-train.tfrecord-00000-of-00001 — 訓練データ（50,000枚）
- cifar10-test.tfrecord-00000-of-00001 — テストデータ（10,000枚）
- dataset_info.json — データセットのメタ情報
- features.json — 特徴量の定義
- label.labels.txt — クラスラベル（airplane, automobile, ... ship, truck）

---



```python

# 読み込み確認
ds_train = builder.as_dataset(split="train")
ds_test  = builder.as_dataset(split="test")

```

この部分について，CIFAR-10 は tfds 側で最初から train / test の2分割が定義されており、それを名前で指定しているらしい．
元のCIFAR-10の公式分割をそのまま使っているだけ．
**CIFAR10の訓練データ5万枚，テスト1万枚**

```
builder.as_dataset(split="train")  # 訓練データ 50,000枚
builder.as_dataset(split="test")   # テストデータ 10,000枚
```
実体はTFRecordファイルの対応関係で、
```
cifar10-train.tfrecord-00000-of-00001  ← split="train"
cifar10-test.tfrecord-00000-of-00001   ← split="test"
```
というファイル名とそのまま対応している．TFRecordファイルとは