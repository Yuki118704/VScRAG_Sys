"""
大規模データをデータベースに追加するスクリプト
ファイルまたはフォルダを指定してドキュメントを追加できます

使い方:
  py add_to_db.py ファイル名.md
  py add_to_db.py フォルダ名
"""

import sys
import os
from pathlib import Path
from RAG_Sys.vector_db import VectorDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ========== チャンク化設定 ==========
CHUNK_SIZE = 1000           # 1チャンクの最大文字数
CHUNK_OVERLAP = 200         # チャンク間のオーバーラップ文字数
ENABLE_CHUNKING = True      # チャンク化を有効にする（False = 見出しのみで分割）
# ====================================


def read_text_file(file_path: Path) -> str:
    """テキストファイルを読み込む"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  ファイル読み込みエラー: {file_path} - {e}")
        return None


def split_large_chunks(text: str, source: str, max_size: int = CHUNK_SIZE) -> list[tuple[str, str]]:
    """
    大きなテキストを適切なサイズに分割
    
    Args:
        text: 分割するテキスト
        source: ソース情報
        max_size: 最大チャンクサイズ
        
    Returns:
        (テキスト, ソース) のリスト
    """
    if not ENABLE_CHUNKING or len(text) <= max_size:
        return [(text, source)]
    
    # LangChainのテキスト分割器を使用
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    # チャンク数が1つなら番号を付けない
    if len(chunks) == 1:
        return [(chunks[0], source)]
    
    # 複数チャンクの場合は番号を付ける
    result = []
    for i, chunk in enumerate(chunks, 1):
        chunk_source = f"{source} (part {i}/{len(chunks)})"
        result.append((chunk, chunk_source))
    
    return result


def split_markdown_by_sections(content: str, source_file: str) -> list[tuple[str, str]]:
    """Markdownを見出しごとに分割"""
    sections = []
    current_section = []
    current_title = "導入"
    
    for line in content.split('\n'):
        if line.startswith('## '):
            # 前のセクションを保存
            if current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append((section_text, f"{source_file} - {current_title}"))
            
            # 新しいセクション開始
            current_title = line[3:].strip()
            current_section = [line]
        else:
            current_section.append(line)
    
    # 最後のセクションを保存
    if current_section:
        section_text = '\n'.join(current_section).strip()
        if section_text:
            sections.append((section_text, f"{source_file} - {current_title}"))
    
    return sections


def process_file(file_path: Path) -> list[tuple[str, str]]:
    """ファイルを処理してテキストとソースのペアを返す"""
    print(f"  📄 {file_path.name}")
    
    content = read_text_file(file_path)
    if not content:
        return []
    
    # Markdownファイルの場合はセクションごとに分割
    if file_path.suffix.lower() in ['.md', '.markdown']:
        sections = split_markdown_by_sections(content, file_path.name)
        
        # 各セクションをさらにチャンク化
        all_chunks = []
        for text, source in sections:
            chunks = split_large_chunks(text, source)
            all_chunks.extend(chunks)
        
        total_chunks = len(all_chunks)
        section_count = len(sections)
        
        if total_chunks > section_count:
            print(f"     → {section_count}セクション → {total_chunks}チャンクに分割")
        else:
            print(f"     → {section_count}セクションに分割")
        
        return all_chunks
    else:
        # その他のファイルはチャンク化して保存
        chunks = split_large_chunks(content, str(file_path.name))
        if len(chunks) > 1:
            print(f"     → {len(chunks)}チャンクに分割")
        return chunks


def main():
    if len(sys.argv) < 2:
        print("使い方: python add_to_db.py <ファイルまたはフォルダ>")
        print("例:")
        print("  python add_to_db.py test_story.md")
        print("  python add_to_db.py ./documents")
        sys.exit(1)
    
    target_path = Path(sys.argv[1])
    
    if not target_path.exists():
        print(f"❌ エラー: {target_path} が見つかりません")
        sys.exit(1)
    
    print("=" * 60)
    print("📚 データベースに大規模データを追加")
    print("=" * 60)
    print(f"チャンク化設定: {'有効' if ENABLE_CHUNKING else '無効'}")
    if ENABLE_CHUNKING:
        print(f"  - チャンクサイズ: {CHUNK_SIZE}文字")
        print(f"  - オーバーラップ: {CHUNK_OVERLAP}文字")
    
    # ファイルを収集
    files_to_process = []
    
    if target_path.is_file():
        files_to_process.append(target_path)
    elif target_path.is_dir():
        # サポートする拡張子
        extensions = ['.txt', '.md', '.markdown', '.rst']
        for ext in extensions:
            files_to_process.extend(target_path.glob(f'**/*{ext}'))
    
    if not files_to_process:
        print("❌ 処理対象のファイルが見つかりません")
        sys.exit(1)
    
    print(f"\n📂 {len(files_to_process)}個のファイルを処理します\n")
    
    # ファイルを処理
    all_texts = []
    all_sources = []
    
    for file_path in files_to_process:
        sections = process_file(file_path)
        for text, source in sections:
            all_texts.append(text)
            all_sources.append(source)
    
    print(f"\n✓ 合計 {len(all_texts)} 件のドキュメントを準備完了")
    
    # データベースに追加
    print("\n🔧 データベースを初期化中...")
    
    # MCPサーバーと同じ場所にデータベースを作成
    db_path = Path(__file__).parent / "RAG_Sys" / "mcp_faiss_db"
    
    db = VectorDatabase(
        collection_name="copilot_rag",
        persist_directory=str(db_path)
    )
    
    print(f"💾 {len(all_texts)} 件のドキュメントを追加中...")
    
    # バッチ処理（10件ずつ）
    batch_size = 10
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_sources = all_sources[i:i+batch_size]
        
        metadatas = [{"source": src} for src in batch_sources]
        db.add_texts(batch_texts, metadatas=metadatas)
        
        processed = min(i + batch_size, len(all_texts))
        print(f"  進捗: {processed}/{len(all_texts)} 件完了")
    
    print("\n" + "=" * 60)
    print("✅ 完了！")
    print(f"   追加されたドキュメント: {len(all_texts)} 件")
    print("=" * 60)


if __name__ == "__main__":
    main()
