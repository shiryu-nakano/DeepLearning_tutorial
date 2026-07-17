# 

準備；
事前に以下を実行しておく必要があるかもしれません．
```
uv add setuptools
```


実行手順：
```
uv run python tutorial.py -F ./output/ with message="world" num_steps=3
```


引数の説明：
- [ ] ローカルで修正しておくこと
引数	説明
-F <dir>	FileStorageObserver の保存先ディレクトリを指定。これがないと cout.txt 等は保存されない
with	この後ろに @ex.config で定義したパラメータの上書きを記述するキーワード
key=value	with 以降に書くパラメータ上書き。ネストは sgld_config.epsilon=2e-7 のようにドット区切り
--comment <text>	実行へのメモ。run.json の meta.comment に保存される
-m <host>	MongoDB Observer の接続先（今回は不使用）



UserWarning: pkg_resources is deprecated as an API. のことですね。

これは Sacred 0.8.5 が内部で pkg_resources（古い setuptools のAPI）を使っているために出る警告で、動作には影響しません。

根本的に消すには Sacred 側のコードを修正するか、より新しいバージョンの Sacred が必要ですが、0.8.5 は2022年以降メンテされていないため修正版はありません。

無視して問題ありません。
