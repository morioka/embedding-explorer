# Embedding Explorer

Streamlit + Plotly + PCA/UMAP + NumPy/SQLite Vec1 によるローカル埋め込み探索アプリです。
画像モデルは必要になった時だけ読み込むオプション機能です。

## 1. インストール

このチェックアウトでは pyenv の `work` 環境を使用します。

```bash
cd /home/morioka/embedding-explorer
pyenv activate work
python -m pip install -r requirements.txt
```

モデルを使う場合は、追加でインストールします。

```bash
python -m pip install -r requirements-model.txt
```

## 2. モデルなしのデモ

デモデータを生成して起動します。

```bash
PYTHONPATH=. python scripts/make_demo.py
streamlit run app.py
```

デモは72件の人工画像と、48次元の合成ベクトルを使用します。実際の画像埋め込みではありません。
`data/` はデモ専用です。

## 3. SQLite Vec1 の準備

このプロジェクトは公式SQLite Vec1を使用します。第三者製の `sqlite-vec` は使用しません。

```bash
pyenv activate work
python -m pip install -r requirements.txt
./scripts/build_vec1.sh
export VEC1_EXTENSION="$PWD/build/vec1.so"
```

ビルドスクリプトは公式 `vec1.c` とSQLite 3.51系ヘッダーを取得し、`work` 環境の
`pysqlite3-binary` に対応した共有ライブラリを作成します。最後に `vec1_info()` を実行して検証します。

CPUがAVX2/FMAに対応していない場合は、`scripts/build_vec1.sh` の `-mavx2 -mfma` を削除してビルドしてください。

## 4. EmbeddingGemma 2 のダウンロード

使用するモデルは [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) です。
テキストと画像を同じ768次元の埋め込み空間に変換できます。

モデルはアプリが自動取得しないため、先にローカルへダウンロードします。通常、Hugging Faceの認証は不要です。

```bash
pyenv activate work
python -m pip install -r requirements-model.txt
python -m pip install -U huggingface_hub

mkdir -p models
hf download google/embeddinggemma-2 \
  --local-dir "$PWD/models/embeddinggemma-2"
```

またはPythonから取得できます。

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/embeddinggemma-2",
    local_dir="models/embeddinggemma-2",
)
PY
```

モデルディレクトリには、設定、トークナイザー、画像プロセッサ、重みなどが含まれている必要があります。

```bash
test -f models/embeddinggemma-2/config.json
test -f models/embeddinggemma-2/tokenizer.json
```

## 5. 実画像データセットの作成

デモ用の `data/` とGemma2用のデータは必ず分けてください。デモは48次元、Gemma2は768次元です。

```bash
mkdir -p data/gemma2/images
cp /path/to/images/* data/gemma2/images/

PYTHONPATH=. python scripts/build_images.py \
  --model "$PWD/models/embeddinggemma-2" \
  --data data/gemma2
```

起動後、サイドバーで次を指定します。

- Data directory: `data/gemma2`
- Local model directory: `/home/morioka/embedding-explorer/models/embeddinggemma-2`

`SentenceTransformer(..., local_files_only=True)` を使用するため、実行時にモデルを暗黙取得しません。

## 6. Streamlitでの画像追加

1. モデルごとに空のデータディレクトリを用意する。
2. サイドバーでモデルディレクトリとデータディレクトリを指定する。
3. PNG/JPEG/WebPをアップロードする。
4. 「Embed, store and reproject」を押す。

画像は埋め込み生成後、SQLiteの `samples` に保存されます。Vec1はSQLite内の全ベクトルから再構築され、
投影器も再学習されます。既存画像と同じバイト列はSHA-256で重複除外されます。

## 7. データと次元数

SQLiteがデータの正本です。`embeddings.npz` はSQLiteが空の場合の初回取り込みに使用されます。
既存DBがある場合、NPZの変更は自動反映されません。

| データ | 次元 | 用途 |
|---|---:|---|
| `data/` | 48 | 合成デモ |
| `data/gemma2/` | 768 | EmbeddingGemma 2 |

異なるモデル、前処理、次元数のベクトルを同じDBに混在させないでください。

既存NPZを意図的に置き換える場合のみ、次を使用します。

```bash
PYTHONPATH=. python scripts/import_npz.py --data data/gemma2 --replace
```

## 8. 仕組み

- PCA/UMAPは表示用の2次元投影です。
- k-NN検索は元の高次元ベクトルで実行します。
- SQLite Vec1はflat cosine検索を使用します。
- SQLiteの `samples` とVec1の `rowid` でメタデータとベクトルを対応付けます。
- 新しいクエリは保存済み投影器の `transform()` だけで2次元へ配置します。
- 投影座標はSQLite、投影器は `projection_models/` に保存されます。

## 9. テスト

```bash
VEC1_EXTENSION="$PWD/build/vec1.so" python -m pytest -q
```

UMAP、Streamlit UI、実モデル推論は環境依存です。Vec1関連テストは `VEC1_EXTENSION` が設定されている場合に実行されます。

## 参考資料

- [EmbeddingGemma 2 - Hugging Face](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma 2 - Google AI](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [SQLite Vec1](https://sqlite.org/vec1/doc/trunk/doc/vec1.md)
