
##　仮想環境構築（uv）

```
cd tensorflow_tutorial

# pyproject.toml を作成（uvのプロジェクト管理を使う場合）
uv init --python 3.12

# 必要なパッケージを追加（.venv が自動作成され、pyproject.toml/uv.lock に記録される）
uv add tensorflow tensorflow-datasets
uv add importlib-resources

# 実行
uv run python down_load_cifar10.py
# もしくは以下でactivateしてからじっこうしても同じ
source .venv/bin/activate
python down_load_cifar10.py


```


