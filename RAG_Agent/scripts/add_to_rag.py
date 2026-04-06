"""
ドキュメントをRAGベクトルデータベースに追加するスクリプト
ファイルまたはフォルダを指定してドキュメントを追加できます

使い方:
  py add_to_rag.py ファイル名.md
  py add_to_rag.py フォルダ名
  py add_to_rag.py ファイル名.md --chunk_size 500
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, COLLECTION_NAME, DB_PATH


# ========== チャンク化設定 ==========
CHUNK_SIZE = 1000           # 1チャンクの最大文字数
CHUNK_OVERLAP = 200         # チャンク間のオーバーラップ文字数
ENABLE_CHUNKING = True      # チャンク化を有効にする
# ====================================


def read_text_file(file_path: Path) -> str:
    """テキストファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  ファイル読み込みエラー: {file_path} - {e}")
        return None


def split_large_chunks(text: str, source: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[tuple[str, str]]:
    """
    大きなテキストを適切なサイズに分割
    
    Args:
        text: 分割するテキスト
        source: ソース情報
        chunk_size: チャンクサイズ
        chunk_overlap: オーバーラップサイズ
        
    Returns:
        (テキスト, ソース) のリスト
    """
    if not ENABLE_CHUNKING or len(text) <= chunk_size:
        return [(text, source)]
    
    # LangChainのテキスト分割器を使用
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    if len(chunks) == 1:
        return [(chunks[0], source)]
    
    result = []
    for i, chunk in enumerate(chunks, 1):
        chunk_source = f"{source} (part {i}/{len(chunks)})"
        result.append((chunk, chunk_source))
    
    return result


def split_json_by_items(content: str, source_file: str) -> list[tuple[str, str]]:
    """JSONの配列要素を1件ずつドキュメントに変換する"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"     ⚠️  JSON解析エラー: {e}")
        return [(content, source_file)]

    # トップレベルが配列の場合はそのまま使用
    # トップレベルがオブジェクトの場合は配列値を探す
    items = None
    array_key = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                items = value
                array_key = key
                break

    if items is None:
        # 配列が見つからない場合はオブジェクト全体を1ドキュメントとして扱う
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return [(text, source_file)]

    results = []
    for i, item in enumerate(items, 1):
        # 各フィールドを「フィールド名: 値」の形式で自然文に変換
        lines = []
        for key, value in item.items():
            if isinstance(value, list):
                value_str = "、".join(
                    v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    for v in value
                )
            else:
                value_str = str(value)
            lines.append(f"{key}: {value_str}")
        text = "\n".join(lines)

        # ソース名: ファイル名 - 配列キー[インデックス] or ファイル名[インデックス]
        item_id = item.get("id") or item.get("name") or item.get("title") or str(i)
        if array_key:
            source = f"{source_file} - {array_key}[{item_id}]"
        else:
            source = f"{source_file}[{item_id}]"

        results.append((text, source))

    return results


def split_markdown_by_sections(content: str, source_file: str) -> list[tuple[str, str]]:
    """Markdownを見出しごとに分割"""
    sections = []
    current_section = []
    current_title = "導入"
    
    for line in content.split('\n'):
        if line.startswith('## '):
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append((section_text, f"{source_file} - {current_title}"))
            
            current_title = line[3:].strip()
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            sections.append((section_text, f"{source_file} - {current_title}"))
    
    return sections


def process_file(file_path: Path, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[tuple[str, str]]:
    """ファイルを処理してテキストとソースのペアを返す"""
    print(f"  📄 {file_path.name}")
    
    content = read_text_file(file_path)
    if not content:
        return []
    
    if file_path.suffix.lower() in ['.md', '.markdown']:
        sections = split_markdown_by_sections(content, file_path.name)
        
        all_chunks = []
        for text, source in sections:
            chunks = split_large_chunks(text, source, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
        
        total_chunks = len(all_chunks)
        section_count = len(sections)
        
        if total_chunks > section_count:
            print(f"     → {section_count}セクション → {total_chunks}チャンクに分割")
        else:
            print(f"     → {section_count}セクションに分割")
        
        return all_chunks
    elif file_path.suffix.lower() == '.json':
        items = split_json_by_items(content, file_path.name)
        print(f"     → {len(items)}件のアイテムに分割")
        return items
    else:
        chunks = split_large_chunks(content, str(file_path.name), chunk_size, chunk_overlap)
        if len(chunks) > 1:
            print(f"     → {len(chunks)}チャンクに分割")
        return chunks


def add_documents(target_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> dict:
    """
    ドキュメントをデータベースに追加
    
    Args:
        target_path: ファイルまたはフォルダのパス
        chunk_size: チャンクサイズ
        chunk_overlap: オーバーラップサイズ
        
    Returns:
        処理結果の辞書
    """
    path = Path(target_path)
    
    if not path.exists():
        return {
            "status": "error",
            "message": f"パスが見つかりません: {target_path}"
        }
    
    print("=" * 60)
    print("📚 データベースにドキュメントを追加")
    print("=" * 60)
    
    # ファイルを収集
    files_to_process = []
    extensions = ['.txt', '.md', '.markdown', '.rst', '.py', '.json', '.yaml', '.yml']
    
    if path.is_file():
        files_to_process.append(path)
    elif path.is_dir():
        for ext in extensions:
            files_to_process.extend(path.glob(f'**/*{ext}'))
    
    if not files_to_process:
        return {
            "status": "error",
            "message": "処理対象のファイルが見つかりません"
        }
    
    print(f"\n📂 {len(files_to_process)}個のファイルを処理します\n")
    
    # ファイルを処理
    all_texts = []
    all_sources = []
    
    for file_path in files_to_process:
        sections = process_file(file_path, chunk_size, chunk_overlap)
        for text, source in sections:
            all_texts.append(text)
            all_sources.append(source)
    
    print(f"\n✓ 合計 {len(all_texts)} 件のドキュメントを準備完了")
    
    # データベースに追加
    print("\n🔧 データベースを初期化中...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 既存DBがあれば読み込み
    index_path = os.path.join(DB_PATH, f"{COLLECTION_NAME}.faiss")
    vectorstore = None
    if os.path.exists(index_path):
        vectorstore = FAISS.load_local(
            DB_PATH, embeddings, COLLECTION_NAME,
            allow_dangerous_deserialization=True
        )
    
    print(f"💾 {len(all_texts)} 件のドキュメントを追加中...")
    
    # バッチ処理（10件ずつ）
    batch_size = 10
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_sources = all_sources[i:i+batch_size]
        
        metadatas = [{"source": src} for src in batch_sources]
        
        if vectorstore is None:
            vectorstore = FAISS.from_texts(
                texts=batch_texts,
                embedding=embeddings,
                metadatas=metadatas
            )
        else:
            vectorstore.add_texts(
                texts=batch_texts,
                metadatas=metadatas
            )
        
        processed = min(i + batch_size, len(all_texts))
        print(f"  進捗: {processed}/{len(all_texts)} 件完了")
    
    # データベースを保存
    os.makedirs(DB_PATH, exist_ok=True)
    vectorstore.save_local(DB_PATH, COLLECTION_NAME)
    print("💾 データベースを保存しました")
    
    print("\n" + "=" * 60)
    print("✅ 完了！")
    print("=" * 60)
    
    return {
        "status": "success",
        "files_processed": len(files_to_process),
        "documents_added": len(all_texts),
        "file_list": [str(f.name) for f in files_to_process]
    }


def main():
    parser = argparse.ArgumentParser(description="RAGデータベースにドキュメントを追加")
    parser.add_argument("path", help="追加するファイルまたはフォルダのパス")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help=f"チャンクサイズ（デフォルト: {CHUNK_SIZE}）")
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP, help=f"オーバーラップサイズ（デフォルト: {CHUNK_OVERLAP}）")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    
    args = parser.parse_args()
    
    result = add_documents(args.path, args.chunk_size, args.chunk_overlap)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["status"] == "success":
        print(f"\n📊 処理結果:")
        print(f"   ファイル数: {result['files_processed']}")
        print(f"   追加ドキュメント: {result['documents_added']}")


if __name__ == "__main__":
    main()
