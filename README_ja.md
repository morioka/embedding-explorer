# Embedding Explorer

Streamlit + Plotly + PCA/UMAP + NumPy/SQLite Vec1 による完全ローカルのUIです。画像モデルはオプションで、必要になった時だけ遅延ロードします。

## 1. インストール（Ubuntu / WSL2、Python 3.11または3.12推奨）

通常のvenvを使う場合：

```bash
cd embedding-explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

uvを使う場合：`uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`

このチェックアウトで検証済みのローカル環境は、pyenvの `work` 環境です。

```bash
pyenv activate work
python -m pip install -r requirements.txt
```

## 2. モデルをダウンロードしないデモ

```bash
PYTHONPATH=. python scripts/make_demo.py
streamlit run app.py
```

デモには72個の生成グラフィックサムネイルと、48次元の**合成クラスタベクトル**が含まれます。実際の画像埋め込みではありません。
クエリソースで **Existing sample** を選び、PCA/UMAP、k、サムネイル、NumPy/SQLite Vec1を切り替えてパイプラインを確認できます。

## 3. 既存の埋め込みを使う

`data/embeddings.npz` を次の形式で用意します。

- `vectors`: float32の `(N,D)` 配列。有限値、非ゼロ、全て同じ次元であること。
- `labels`: N個の文字列。
- `paths`: N個の文字列。データディレクトリからの相対画像パス（例：`images/cat.png`）。

例：

```python
import numpy as np
np.savez_compressed('data/embeddings.npz', vectors=vectors.astype('float32'), labels=np.array(labels), paths=np.array(paths))
```

任意で `data/manifest.json` に `{"model_id":"some-model-revision", "dimension":768}` を指定できます。

**クエリのアップロード**：D次元の1ベクトルを含む `.npy`、または `vector` キーを持つ `.npz` を使用できます。サンプルベクトルと同じモデル、前処理、次元数を使ってください。データセットは内部でL2正規化されます。

## 4. EmbeddingGemma 2で実画像を使う

使用するモデルは [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) です。テキストと画像を同じ768次元の埋め込み空間へ写像します。
アプリはモデルを自動取得しないため、トークナイザー、画像プロセッサ、pooling、設定、重みを含む完全なローカルモデルディレクトリを先に用意してください。

モデル依存関係を `work` 環境へインストールします。

```bash
pyenv activate work
pip install -r requirements-model.txt
```

Hugging Faceからモデルをダウンロードします。通常、このモデルの取得にHugging Face認証は必要ありません。

```bash
pip install -U huggingface_hub
mkdir -p models
hf download google/embeddinggemma-2 \
  --local-dir "$PWD/models/embeddinggemma-2"
```

Pythonから取得することもできます。

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/embeddinggemma-2",
    local_dir="models/embeddinggemma-2",
)
PY
```

ディレクトリが完全であることを確認します。

```bash
test -f models/embeddinggemma-2/config.json
test -f models/embeddinggemma-2/tokenizer.json
find models/embeddinggemma-2 -maxdepth 1 -type f | sort
```

画像インデックス作成とサイドバーには、ダウンロードしたディレクトリを指定します。

```bash
# 少なくとも3個の .png/.jpg/.webp を data/gemma2/images/ に置く
PYTHONPATH=. python scripts/build_images.py \
  --model "$PWD/models/embeddinggemma-2" --data data/gemma2
streamlit run app.py
```

サイドバーの **Data directory** には `data/gemma2`、**Local model directory** には同じモデルの絶対パスを指定してください。
`SentenceTransformer(..., local_files_only=True)` により暗黙のモデル取得を防止します。実行環境に適したPyTorchが必要です。
CPUでも推論できますが低速です。付属の `data/` は48次元の合成デモであり、768次元のGemma2データセットに再利用してはいけません。

## 5. 設計の詳細

- 投影器は**コーパスだけ**にfitします。クエリは `transform()` を使い、埋め込みマップを再fitしません。
- UMAPとPCAは、単位長に正規化したベクトルを処理します。
- k-NNは**元のベクトル**を検索します。類似度は `1 - cosine_distance` です。SQLite Vec1はcosine距離のflat検索に設定されています。
- SQLiteを正本とします。NPZはDBが空の場合、または明示的に置き換える場合だけインポートします。一括追加はトランザクションで行い、各バッチ後にSQLiteの全埋め込みからVec1 flatインデックスを再構築します。
- サンプル点にはScatterglを使い、サムネイルにはPlotlyのlayout imageを使います。サムネイル数には上限があります。点を選びにくい場合はサムネイルをオフにしてください。
- 合成ベクトルを含め、異なるモデル空間を同じ検索インデックスに混在させてはいけません。データの出所は外部でも確認してください。
- 特にUMAPで新しいクエリをtransformした場合、投影上の距離と近傍順位は一致しないことがあります。
- `data_dir` サイドバー入力は信頼できるローカルユーザー向けです。認証付きマルチテナントサービスではありません。

## 6. 確認

```bash
pip install pytest
VEC1_EXTENSION="$PWD/build/vec1.so" python -m pytest -q
```

コア、投影、SQLiteのユニットテストを用意しています。Streamlit UIと実際のEmbeddingGemma 2推論はエンドツーエンドテストの対象外です。

## 参考資料

- https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
- https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers
- https://huggingface.co/google/embeddinggemma-2
- https://sqlite.org/vec1/doc/trunk/doc/vec1.md
- https://umap-learn.readthedocs.io/en/latest/transform.html

## SQLiteファーストの永続化（更新後の設計）

`data/vectors.db` はサンプルのラベル、画像パス、float32ベクトルの正本です。`sample_vectors` は公式SQLite Vec1の `vec1(vector)` インデックスで、flat cosine NNに設定されています。初回起動時、DBが空の場合に限り既存の `embeddings.npz` をインポートします。

その後の起動ではサンプルをSQLiteから直接読み込み、NPZの変更は無視します。サンプルを意図的に置き換える場合は、次を実行します。

```bash
PYTHONPATH=. python scripts/import_npz.py --data data/gemma2 --replace
```

PCA/UMAPの学習座標はSQLiteの `projections` テーブルに保存されます。学習済み変換器は `data/projection_models/<key>.pkl` に保存され、両方が存在すると再利用されます。不整合または欠損がある場合は再学習します。

各投影方式・パラメータの初回実行時に学習・保存されます。新しいクエリベクトルは常にfit済みモデルの `transform()` を使い、サンプル投影を再fitしません。

**セキュリティ:** 投影モデルはPython pickleを使うため、信頼できないファイルを読み込んではいけません。このアプリが作成したローカルキャッシュだけを使用してください。SQLiteのベクトル配列と投影座標にはpickleのデシリアライズは不要です。

`import_npz.py --replace` による置き換えではデータセット世代が進み、古い投影キーが無効になります。キャッシュディレクトリを削除するか、対象キーの `projections` 行を削除すると再学習できます。

## Streamlitでの画像一括追加

1. 選択するモデル用の**空のデータディレクトリ**から開始します。例：`data/my_photos`。付属の `data/` は合成デモベクトルを含むため、実画像ベクトルと混在させてはいけません。
2. `requirements-model.txt` をインストールし、完全なローカルEmbeddingGemma 2モデルディレクトリをサイドバーで指定します。
3. 「Add sample images (batch)」へPNG/JPEG/WebPを複数ドロップし、「Embed, store and reproject」を押します。
4. アプリは画像をデコードし、埋め込みをSQLiteの `samples` に追加し、SQLite全件からVec1を再構築し、データセット世代を進め、選択中のPCA/UMAPを再fitし、座標と変換器を保存して画面を再描画します。他の投影方式は次に選択した時に再fitされます。
5. 同じバイト列の重複アップロードはSHA-256で無視されます。画像は `images/uploaded/` にコピーされます。モデル由来の検査により、異なるモデルパスのデータを同じデータセットへ追加できません。由来情報のない既存ベクトルには別のデータディレクトリを使ってください。

埋め込み推論はStreamlitプロセス内で同期的に実行されます。これは単純なローカル処理であり、大量アップロード用のバックグラウンドジョブではありません。バッチ中にプロセスを終了すると未使用の画像ファイルが残る可能性がありますが、SQLiteへ途中状態のエントリは追加されません。明示的なボタンを使うため、画像をドロップしただけでは推論を実行しません。

## 公式SQLite Vec1のインストール（必須、sqlite-vecではない）

Vec1は単一のCソース `vec1.c` として配布される公式SQLite拡張です。第三者製の `sqlite-vec` とは無関係で、スキーマとSQL APIも異なります。

このプロジェクトは `work` pyenv環境を使用します。Ubuntu/WSL x86-64では次を実行します。

```bash
cd /home/morioka/embedding-explorer
pyenv activate work
python -m pip install -r requirements.txt
./scripts/build_vec1.sh
export VEC1_EXTENSION="$PWD/build/vec1.so"
```

ビルドスクリプトは公式ソースを取得し、AVX2/FMAでコンパイルし、`pysqlite3-binary` に含まれるSQLiteランタイムへリンクし、`SELECT vec1_info()` を検証します。結果には `Vec1 0.7` が含まれます。

同じシェルでアプリを起動します。

```bash
pyenv activate work
export VEC1_EXTENSION="$PWD/build/vec1.so"
streamlit run app.py
```

`explorer.store` は `build/vec1.so` も自動検索するため、プロジェクトルートから実行する場合は環境変数を省略できます。ただし、スクリプトやサービスでは明示的に設定することを推奨します。

標準のコンパイルフラグはAVX2/FMA対応のx86-64 CPUを必要とします。非対応ハードウェアでは `scripts/build_vec1.sh` の `-mavx2 -mfma` を削除してください。PythonのSQLite実装は拡張ロードに対応している必要があり、`work` 環境では `pysqlite3-binary` がこれを提供します。

検索には公式SQL APIを使います。

```sql
CREATE VIRTUAL TABLE sample_vectors USING vec1(vector);
INSERT INTO sample_vectors(rowid, vector) VALUES(123, :float32_blob);
INSERT INTO sample_vectors(cmd, arg)
  VALUES('rebuild', '{index:"flat", distance:"cos"}');
SELECT rowid, vec1_cos_distance(:query_blob, vector) AS distance
  FROM sample_vectors(:query_blob, '{k:10}') ORDER BY distance;
```

標準設定は**厳密なflat NN**であり、ANNではありません。およそ5,000埋め込み以下の探索に適しています。ANNを使う場合は、明示的な `vec1_train()`、学習済みモデルを指定した `rebuild`、候補の再ランキングが必要です。このパッケージはANNを暗黙には有効化しません。Vec1は `sqlite-vec` ではなく、既存のvec0データベースはそのまま再利用できません。NPZから新しいVec1 DBへ再インポートしてください。
