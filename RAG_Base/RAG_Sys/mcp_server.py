"""
VScRAG - VSCode RAG System
GitHub CopilotにRAG機能を提供するMCPサーバ
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

from vector_db import VectorDatabase


# グローバルなデータベースインスタンス
db = None


def check_model_cache(model_name: str = "intfloat/multilingual-e5-base") -> bool:
    """モデルがキャッシュに存在するかチェック"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    return model_cache.exists()


async def initialize_rag():
    """RAGシステムを初期化"""
    global db
    
    import sys
    
    # モデルの存在確認
    model_name = "intfloat/multilingual-e5-base"
    if not check_model_cache(model_name):
        print("=" * 60, file=sys.stderr)
        print("[VScRAG] ⚠️  警告: 埋め込みモデルが見つかりません", file=sys.stderr)
        print(f"[VScRAG] モデル: {model_name}", file=sys.stderr)
        print("[VScRAG]", file=sys.stderr)
        print("[VScRAG] 初回実行時は以下を実行してください:", file=sys.stderr)
        print("[VScRAG]   python init_model.py", file=sys.stderr)
        print("[VScRAG]", file=sys.stderr)
        print("[VScRAG] ダウンロードを開始します（数分かかります）...", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
    
    print("[VScRAG] 初期化を開始します...", file=sys.stderr)
    
    db = VectorDatabase(
        collection_name="copilot_rag",
        persist_directory="./mcp_faiss_db",
        embedding_model=model_name
    )
    
    print("[VScRAG] 初期化が完了しました", file=sys.stderr)
    return {"status": "initialized"}


# MCPサーバのインスタンスを作成
server = Server("vscrag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """利用可能なツールのリストを返す"""
    return [
        Tool(
            name="search_documents",
            description="質問に関連するドキュメントをベクトルデータベースから検索します。ユーザーがデータ追加を要求した場合は、ターミナルで「python add_to_db.py ファイル名」を実行するよう案内してください。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "返す結果の数（デフォルト: 3）"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_db_stats",
            description="RAGデータベースの統計情報を取得します。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """ツールを実行"""
    global db
    
    import sys
    
    print(f"[VScRAG] ツール '{name}' を実行中...", file=sys.stderr)
    
    try:
        if name == "search_documents":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 3)
            
            print(f"[VScRAG] 検索パラメータ: query='{query}', top_k={top_k}", file=sys.stderr)
            
            # 検索実行
            results = db.similarity_search_with_score(query, k=top_k)
            
            print(f"[VScRAG] ツール '{name}' の実行が完了しました ({len(results)}件)", file=sys.stderr)
            
            # 結果をフォーマット
            search_results = []
            for i, (doc, score) in enumerate(results, 1):
                search_results.append({
                    "rank": i,
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata
                })
            
            result = {
                "query": query,
                "results": search_results,
                "total_found": len(search_results)
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        
        elif name == "get_db_stats":
            # データベースの統計情報を取得
            import os
            import pickle
            
            persist_directory = "./mcp_faiss_db"
            collection_name = "copilot_rag"
            pkl_file = os.path.join(persist_directory, f"{collection_name}.pkl")
            faiss_file = os.path.join(persist_directory, f"{collection_name}.faiss")
            
            stats = {
                "collection_name": collection_name,
                "persist_directory": persist_directory,
                "faiss_file_exists": os.path.exists(faiss_file),
                "pkl_file_exists": os.path.exists(pkl_file)
            }
            
            # 実際のベクトルストアからドキュメント数を取得
            if db and db.vectorstore:
                try:
                    stats["total_documents"] = db.vectorstore.index.ntotal
                except Exception as e:
                    stats["total_documents"] = f"取得エラー: {str(e)}"
            else:
                stats["total_documents"] = 0
            
            # ファイルサイズ情報
            if os.path.exists(faiss_file):
                stats["faiss_file_size_kb"] = round(os.path.getsize(faiss_file) / 1024, 2)
            if os.path.exists(pkl_file):
                stats["pkl_file_size_kb"] = round(os.path.getsize(pkl_file) / 1024, 2)
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, ensure_ascii=False, indent=2)
            )]
        
        else:
            return [TextContent(
                type="text",
                text=f"不明なツール: {name}"
            )]
    
    except Exception as e:
        error_result = {
            "error": str(e),
            "tool": name
        }
        return [TextContent(
            type="text",
            text=json.dumps(error_result, ensure_ascii=False, indent=2)
        )]


async def main():
    """MCPサーバを起動"""
    import sys
    print("[VScRAG] MCPサーバを起動しています...", file=sys.stderr)
    
    # サーバー起動時に初期化（モデルを事前読み込み）
    await initialize_rag()
    
    async with stdio_server() as (read_stream, write_stream):
        print("[VScRAG] サーバーが準備完了しました", file=sys.stderr)
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
