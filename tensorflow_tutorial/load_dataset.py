"""
キャッシュ済みの CIFAR-10 からサンプリングするスクリプト。
"""

import os
import tensorflow_datasets as tfds
import tensorflow as tf

# download_cifar10.py と同じ保存先を指定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "tensorflow_datasets")

# ダウンロードせずキャッシュから読み込む
builder = tfds.builder("cifar10", data_dir=DATA_DIR)
ds_train = builder.as_dataset(split="train", as_supervised=True)
ds_test  = builder.as_dataset(split="test",  as_supervised=True)

# --- サンプリング例 ---

# 1. 先頭N件だけ取る
N = 1000
ds_small = ds_train.take(N)
ds_train.take(N).cache()
print(f"take({N}): {ds_small.cardinality().numpy()} サンプル")

# 2. シャッフルしてから取る（再現性のためseedを固定）
ds_shuffled: tf.data.Dataset = ds_train.shuffle(buffer_size=10000, seed=42).take(N)
print(f"shuffle + take({N}): {ds_shuffled.cardinality().numpy()} サンプル")

# 3. バッチ化
BATCH_SIZE = 32
ds_batched: tf.data.Dataset = ds_train.shuffle(10000, seed=42).batch(BATCH_SIZE)
print(f"batch({BATCH_SIZE}): {ds_batched.cardinality().numpy()} バッチ")

# 4. train を train/val に分割（8:2）
TRAIN_SIZE = 40000
ds_tr  = ds_train.take(TRAIN_SIZE)
ds_val = ds_train.skip(TRAIN_SIZE)
print(f"train: {ds_tr.cardinality().numpy()}, val: {ds_val.cardinality().numpy()}")

# --- 中身の確認 ---
for image, label in ds_small.take(1):
    print(f"\nimage shape : {image.shape}")   # (32, 32, 3)
    print(f"image dtype : {image.dtype}")    # uint8
    print(f"label       : {label.numpy()}")  # 0~9
    

"""
キャッシュ済みの CIFAR-10 をサンプリングして可視化するスクリプト。
"""

import os
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "tensorflow_datasets")

# ラベル名
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# キャッシュから読み込み
builder = tfds.builder("cifar10", data_dir=DATA_DIR)
ds_train = builder.as_dataset(split="train", as_supervised=True)

# シャッフルして32枚サンプリング
samples = list(ds_train.shuffle(10000, seed=42).take(32))

# 4x8 グリッドで表示
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("CIFAR-10 samples", fontsize=16)

for ax, (image, label) in zip(axes.flat, samples):
    ax.imshow(image.numpy())
    ax.set_title(CLASS_NAMES[label.numpy()], fontsize=8)
    ax.axis("off")

plt.tight_layout()

# 保存
OUT_PATH = os.path.join(BASE_DIR, "cifar10_samples.png")
plt.savefig(OUT_PATH, dpi=150)
print(f"保存しました: {OUT_PATH}")

plt.show()    