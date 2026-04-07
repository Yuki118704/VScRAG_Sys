# RAG Agent

VS Code GitHub Copilot Agent向けのRAG（Retrieval-Augmented Generation）システムです。
FAISSベクトルDBを使ってドキュメントの検索・追加を行います。

## セットアップ

```powershell
# 依存ライブラリを一括インストール（リポジトリルートで実行）
pip install -r requirements.txt
```

## ディレクトリ構成

```
VScRAG_Sys/
├── requirements.txt            # 依存ライブラリ一覧
└── RAG_Agent/
    ├── config.py               # 埋め込みモデル・DB設定
    ├── faiss_db/               # ベクトルDB保存先
    ├── .github/agents/
    │   └── rag.agent.md        # Copilot Agent定義
    └── scripts/
        ├── search_rag.py       # 検索
        ├── add_to_rag.py       # データ追加
        └── get_stats.py        # 統計確認
```

## 使い方

### 1. Copilot Agentから使う（推奨）

VS Codeのチャットでエージェントピッカーから **「RAG Agent」** を選択し、以下のように話しかけます。

```
RAGデータベースから○○を検索して
ナレッジベースにこのファイルを追加して
ベクトルDBの状態を確認して
```

### 2. スクリプトを直接実行する

```powershell
cd RAG_Agent/scripts

# ドキュメントを検索
py search_rag.py "検索クエリ"
py search_rag.py "検索クエリ" --top_k 5

# ドキュメントを追加（ファイルまたはフォルダ指定）
py add_to_rag.py path/to/document.md
py add_to_rag.py path/to/folder
py add_to_rag.py path/to/document.md --chunk_size 500

# DBの統計情報を確認
py get_stats.py
py get_stats.py --json
```

## 対応ファイル形式

`.md` `.txt` `.py` `.json` `.yaml` `.yml`

## 設定 (config.py)

| 設定項目 | デフォルト値 | 説明 |
|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | 埋め込みモデル |
| `EMBEDDING_DEVICE` | `cpu` | 実行デバイス |
| `COLLECTION_NAME` | `copilot_rag` | DBコレクション名 |
| `DB_PATH` | `RAG_Agent/faiss_db/` | DB保存先パス |
