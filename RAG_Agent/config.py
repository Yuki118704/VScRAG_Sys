"""
RAG Agent 設定ファイル
"""

import os

# 埋め込みモデル設定
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DEVICE = "cpu"

# データベース設定
COLLECTION_NAME = "copilot_rag"
DB_PATH = os.path.join(os.path.dirname(__file__), 'faiss_db')
