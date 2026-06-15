"""
CIFAR-10 を tensorflow_datasets でダウンロード・キャッシュするスクリプト。
サーバ上の ~/tensorflow_datasets/ と同じディレクトリ構造をローカルに再現する。
"""

import os
import tensorflow_datasets as tfds

# 保存先ディレクトリ（このスクリプトからの相対パス）
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "tensorflow_datasets")

print(f"データ保存先: {DATA_DIR}")
print("CIFAR-10 のダウンロードとTFRecord変換を開始します...")


# ダウンロード・TFRecord変換・キャッシュ保存
builder = tfds.builder("cifar10", data_dir=DATA_DIR)
builder.download_and_prepare()


print("\n変換完了。データセットを読み込みます...")

# 読み込み確認
ds_train = builder.as_dataset(split="train")
ds_test  = builder.as_dataset(split="test")

print(f"train サンプル数: {ds_train.cardinality().numpy()}")
print(f"test  サンプル数: {ds_test.cardinality().numpy()}")

print("\n保存されたファイル一覧:")
version_dir = os.path.join(DATA_DIR, "cifar10", "3.0.2")
for f in sorted(os.listdir(version_dir)):
    path = os.path.join(version_dir, f)
    size = os.path.getsize(path)
    print(f"  {f}  ({size / 1024 / 1024:.1f} MB)")

print("\n完了。次回以降は download_and_prepare() をスキップしてキャッシュが使われます。")
