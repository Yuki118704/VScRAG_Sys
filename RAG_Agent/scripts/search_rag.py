"""
RAGベクトルデータベースから関連ドキュメントを検索するスクリプト

使い方:
  py search_rag.py "検索クエリ"
  py search_rag.py "検索クエリ" --top_k 5
"""

import sys
import os
import argparse
import json

from vector_db import get_embeddings, load_vectorstore, DB_PATH


def search_documents(query: str, top_k: int = 3) -> dict:
    """
    ベクトルDBからドキュメントを検索
    
    Args:
        query: 検索クエリ
        top_k: 返す結果の数
        
    Returns:
        検索結果の辞書
    """
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    
    if vectorstore is None:
        return {
            "status": "no_results",
            "message": "データベースが存在しません",
            "query": query
        }
    
    # 検索実行
    results = vectorstore.similarity_search_with_score(query=query, k=top_k)
    
    if not results:
        return {
            "status": "no_results",
            "message": "関連するドキュメントが見つかりませんでした",
            "query": query
        }
    
    # 結果を整形
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata
        })
    
    return {
        "status": "success",
        "query": query,
        "count": len(formatted_results),
        "results": formatted_results
    }


def main():
    parser = argparse.ArgumentParser(description="RAGベクトルDBから検索")
    parser.add_argument("query", help="検索クエリ")
    parser.add_argument("--top_k", type=int, default=3, help="返す結果の数（デフォルト: 3）")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    
    args = parser.parse_args()
    
    result = search_documents(args.query, args.top_k)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["status"] == "no_results":
            print(f"\n❌ {result['message']}")
        else:
            print(f"\n🔍 検索クエリ: \"{result['query']}\"")
            print(f"📊 結果: {result['count']}件\n")
            print("=" * 60)
            
            for i, item in enumerate(result["results"], 1):
                print(f"\n【結果 {i}】スコア: {item['score']:.4f}")
                if item["metadata"]:
                    source = item["metadata"].get("source", "不明")
                    print(f"📁 ソース: {source}")
                print("-" * 40)
                print(item["content"][:500])
                if len(item["content"]) > 500:
                    print("... (省略)")
                print()


if __name__ == "__main__":
    main()
