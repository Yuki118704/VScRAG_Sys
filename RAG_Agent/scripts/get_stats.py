"""
RAGベクトルデータベースの統計情報を取得するスクリプト

使い方:
  py get_stats.py
  py get_stats.py --json
"""

import sys
import os
import argparse
import json
from pathlib import Path

from vector_db import get_embeddings, load_vectorstore, DB_PATH


def get_db_stats() -> dict:
    """
    データベースの統計情報を取得
    
    Returns:
        統計情報の辞書
    """
    db_path = DB_PATH
    
    # FAISSファイルの存在確認
    faiss_file = os.path.join(db_path, "copilot_rag.faiss")
    pkl_file = os.path.join(db_path, "copilot_rag.pkl")
    
    if not os.path.exists(faiss_file):
        return {
            "status": "empty",
            "message": "データベースが存在しません",
            "db_path": db_path
        }
    
    # ファイルサイズを取得
    faiss_size = os.path.getsize(faiss_file) if os.path.exists(faiss_file) else 0
    pkl_size = os.path.getsize(pkl_file) if os.path.exists(pkl_file) else 0
    
    # データベースに接続
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    
    # ドキュメント数を取得
    doc_count = 0
    sources = set()
    
    if vectorstore is not None:
        # FAISSのインデックスからドキュメント数を取得
        doc_count = vectorstore.index.ntotal
        
        # docstoreからソース情報を収集
        try:
            for doc_id in vectorstore.docstore._dict:
                doc = vectorstore.docstore._dict[doc_id]
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source = doc.metadata['source']
                    # パート番号を除去してユニークなソースを集計
                    base_source = source.split(' (part ')[0]
                    sources.add(base_source)
        except Exception:
            pass
    
    return {
        "status": "success",
        "db_path": db_path,
        "document_count": doc_count,
        "unique_sources": len(sources),
        "source_list": sorted(list(sources)),
        "file_sizes": {
            "faiss_index": f"{faiss_size / 1024:.1f} KB",
            "docstore": f"{pkl_size / 1024:.1f} KB",
            "total": f"{(faiss_size + pkl_size) / 1024:.1f} KB"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="RAGデータベースの統計情報を取得")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    
    args = parser.parse_args()
    
    result = get_db_stats()
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print("📊 RAGデータベース統計情報")
        print("=" * 60)
        
        if result["status"] == "empty":
            print(f"\n❌ {result['message']}")
            print(f"   パス: {result['db_path']}")
        else:
            print(f"\n📁 データベースパス: {result['db_path']}")
            print(f"\n📈 ドキュメント数: {result['document_count']} 件")
            print(f"📂 ユニークソース: {result['unique_sources']} 件")
            
            print(f"\n💾 ファイルサイズ:")
            print(f"   FAISSインデックス: {result['file_sizes']['faiss_index']}")
            print(f"   Docstore: {result['file_sizes']['docstore']}")
            print(f"   合計: {result['file_sizes']['total']}")
            
            if result['source_list']:
                print(f"\n📋 ソース一覧:")
                for source in result['source_list'][:10]:  # 最大10件表示
                    print(f"   - {source}")
                if len(result['source_list']) > 10:
                    print(f"   ... 他 {len(result['source_list']) - 10} 件")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
