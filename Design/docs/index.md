# VScRAG 設計ドキュメント

VSCode RAGシステムの設計・実装ドキュメントへようこそ。

## 📚 ドキュメント構成

---

## 🔍 システム概要

VScRAGは、GitHub CopilotにRAG（Retrieval-Augmented Generation）機能を提供するシステムです。

### 主要コンポーネント
- **MCPサーバー**: GitHub Copilotとの通信を担当
- **ベクトルDB**: FAISSを使用したドキュメント検索
- **ドキュメント追加**: Markdownファイルの自動チャンク化と登録
- **埋め込み生成**: HuggingFace multilingual-e5-baseモデルによる埋め込み

### 主要機能
- ドキュメント類似度検索
- バッチドキュメント追加
- データベース統計情報取得
