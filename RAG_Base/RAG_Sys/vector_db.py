"""
ベクトルデータベースモジュール
FAISSを使用してドキュメントの保存と検索を行う
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional
import os


class VectorDatabase:
    """FAISSを使用したベクトルデータベースクラス"""
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./faiss_db",
        embedding_model: str = "intfloat/multilingual-e5-base"
    ):
        """
        初期化
        
        Args:
            collection_name: コレクション名
            persist_directory: データベースの保存先ディレクトリ
            embedding_model: 埋め込みモデル名
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.index_path = os.path.join(persist_directory, f"{collection_name}.faiss")
        self.docstore_path = os.path.join(persist_directory, f"{collection_name}.pkl")
        
        import sys
        print(f"[VectorDB] 埋め込みモデルを読み込み中: {embedding_model}", file=sys.stderr)
        print(f"[VectorDB] 初回実行の場合、モデルダウンロードに数分かかることがあります", file=sys.stderr)
        
        # 埋め込みモデルの初期化
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print("[VectorDB] モデルの読み込みが完了しました", file=sys.stderr)
        
        # FAISSの初期化
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """既存のベクトルストアを読み込むか、新規作成"""
        import sys
        if os.path.exists(self.index_path):
            print(f"[VectorDB] 既存のデータベースを読み込み中: {self.persist_directory}", file=sys.stderr)
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    self.collection_name,
                    allow_dangerous_deserialization=True
                )
                print(f"[VectorDB] データベース読み込み完了", file=sys.stderr)
            except Exception as e:
                print(f"[VectorDB] データベースの読み込みに失敗: {e}", file=sys.stderr)
                print("[VectorDB] 新しいデータベースを作成します", file=sys.stderr)
                self.vectorstore = None
        else:
            print("[VectorDB] 新しいデータベースを作成します", file=sys.stderr)
            # 保存ディレクトリを作成
            os.makedirs(self.persist_directory, exist_ok=True)
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """
        テキストをベクトルデータベースに追加
        
        Args:
            texts: 追加するテキストのリスト
            metadatas: メタデータのリスト
            
        Returns:
            追加されたドキュメントのIDリスト
        """
        import sys
        print(f"[VectorDB] {len(texts)}件のテキストを追加中...", file=sys.stderr)
        
        if self.vectorstore is None:
            # 初回の場合は新規作成
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
        else:
            # 既存のベクトルストアに追加
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
        
        # FAISSは自動保存しないので明示的に保存
        self._save()
        
        print(f"[VectorDB] 追加完了", file=sys.stderr)
        return [str(i) for i in range(len(texts))]
    
    def _save(self):
        """ベクトルストアをディスクに保存"""
        import sys
        if self.vectorstore is not None:
            print(f"[VectorDB] データベースを保存中...", file=sys.stderr)
            self.vectorstore.save_local(
                self.persist_directory,
                self.collection_name
            )
            print(f"[VectorDB] 保存完了", file=sys.stderr)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        類似度スコア付きで検索
        
        Args:
            query: 検索クエリ
            k: 返す結果の数
            
        Returns:
            (Document, スコア)のタプルのリスト
        """
        import sys
        if self.vectorstore is None:
            print("データベースが空です", file=sys.stderr)
            return []
        
        print(f"[VectorDB] 検索クエリ: '{query}' (top_k={k})", file=sys.stderr)
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        
        print(f"[VectorDB] 検索結果: {len(results)}件", file=sys.stderr)
        for i, (doc, score) in enumerate(results, 1):
            content_preview = doc.page_content[:50].replace('\n', ' ') + "..." if len(doc.page_content) > 50 else doc.page_content.replace('\n', ' ')
            print(f"[VectorDB]   {i}位: スコア={score:.4f}, 内容='{content_preview}'", file=sys.stderr)
        
        return results
