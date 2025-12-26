# VScRAG 設計ドキュメント

VSCode RAGシステムの設計・実装ドキュメントへようこそ。

## 📚 ドキュメント構成

### [00_StkRA - ステークホルダー要求分析](00_StkRA/00_StkRA_overview.md)
システムのステークホルダーと要求を定義

### [10_SysRA - システム要求分析](10_SysRA/10_SysRA_overview.md)
システム全体の要求を分析

### [20_SysAD - システムアーキテクチャ設計](20_SysAD/20_SysAD_overview.md)
システムアーキテクチャの設計

### [30_SwRA - ソフトウェア要求分析](30_SwRA/30_SwRA_overview.md)
ソフトウェアコンポーネントの要求分析

### [40_SwAD - ソフトウェアアーキテクチャ設計](40_SwAD/40_SwAD_overview.md)
ソフトウェアアーキテクチャの詳細設計

### [50_SwUD - ソフトウェアユニット設計](50_SwUD/50_SwUD_overview.md)
ソフトウェアユニットの詳細設計

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
