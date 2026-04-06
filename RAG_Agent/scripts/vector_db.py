"""
ベクトルデータベース管理モジュール
埋め込みモデルとFAISSベクトルストアの初期化・管理を行う
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional
import os
import sys

# デフォルト設定
DEFAULT_COLLECTION = "copilot_rag"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'faiss_db')


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """埋め込みモデルをロードして返す"""
    print(f"[VectorDB] 埋め込みモデルを読み込み中: {model_name}", file=sys.stderr)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("[VectorDB] モデルの読み込みが完了しました", file=sys.stderr)
    return embeddings


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = DB_PATH,
    collection_name: str = DEFAULT_COLLECTION
) -> Optional[FAISS]:
    """既存のFAISSベクトルストアを読み込む。存在しなければNoneを返す"""
    index_path = os.path.join(persist_directory, f"{collection_name}.faiss")
    if not os.path.exists(index_path):
        print("[VectorDB] データベースが見つかりません", file=sys.stderr)
        return None

    print(f"[VectorDB] データベースを読み込み中: {persist_directory}", file=sys.stderr)
    try:
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            collection_name,
            allow_dangerous_deserialization=True
        )
        print("[VectorDB] データベース読み込み完了", file=sys.stderr)
        return vectorstore
    except Exception as e:
        print(f"[VectorDB] データベースの読み込みに失敗: {e}", file=sys.stderr)
        return None


def save_vectorstore(
    vectorstore: FAISS,
    persist_directory: str = DB_PATH,
    collection_name: str = DEFAULT_COLLECTION
) -> None:
    """FAISSベクトルストアをディスクに保存する"""
    os.makedirs(persist_directory, exist_ok=True)
    print("[VectorDB] データベースを保存中...", file=sys.stderr)
    vectorstore.save_local(persist_directory, collection_name)
    print("[VectorDB] 保存完了", file=sys.stderr)
