# RAG Agent

VS Code GitHub Copilot Agent向けのRAG（Retrieval-Augmented Generation）システムです。
Agentがターミナルでpythonスクリプトを実行してベクトルDBの検索・追加を行います。

## 構成

```
RAG_Agent/
├── .github/
│   └── agents/
│       └── rag.agent.md    # Copilot Agent定義
├── scripts/
│   ├── search_rag.py       # 検索スクリプト
│   ├── add_to_rag.py       # データ追加スクリプト
│   └── get_stats.py        # 統計取得スクリプト
└── README.md
```

## 使い方

### Agentの呼び出し

VS Codeのチャットで `@workspace` を使用するか、Agentピッカーから「RAG Agent」を選択します。

または、以下のような質問をすると自動的にRAG Agentが呼び出されます:
- 「RAGデータベースから○○を検索して」
- 「ナレッジベースに新しいドキュメントを追加して」
- 「ベクトルDBの状態を確認して」

### 直接スクリプトを実行する場合

```powershell
cd RAG_Agent/scripts

# 検索
py search_rag.py "検索したいキーワード"
py search_rag.py "検索クエリ" --top_k 5

# データ追加
py add_to_rag.py path/to/document.md
py add_to_rag.py path/to/folder

# 統計確認
py get_stats.py
```

## 前提条件

- RAG_Baseフォルダの`RAG_Sys`モジュールが必要です
- 必要なPythonパッケージ:
  - langchain-huggingface
  - langchain-community
  - faiss-cpu
  - langchain-text-splitters

## データベースの場所

ベクトルDBは `RAG_Agent/faiss_db/` に保存されます。
RAG_Baseとは独立したデータベースです。
