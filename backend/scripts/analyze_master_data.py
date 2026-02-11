#!/usr/bin/env python3
"""
Script để phân tích và kiểm tra chất lượng của Master Data JSONL.
- Tìm chunk ngắn nhất
- Hiển thị tất cả các chunk chứa bảng
- Hiển thị ngẫu nhiên chunks để kiểm tra chất lượng
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich import box

# Khởi tạo Rich Console với giới hạn chiều rộng
console = Console(width=120)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load file JSONL và trả về list các chunks."""
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def find_shortest_chunks(chunks: List[Dict], top_n: int = 10) -> List[Dict]:
    """Tìm N chunk ngắn nhất."""
    sorted_chunks = sorted(chunks, key=lambda x: len(x['content']))
    return sorted_chunks[:top_n]


def find_longest_chunks(chunks: List[Dict], top_n: int = 10) -> List[Dict]:
    """Tìm N chunk dài nhất."""
    sorted_chunks = sorted(chunks, key=lambda x: len(x['content']), reverse=True)
    return sorted_chunks[:top_n]


def find_chunks_with_tables(chunks: List[Dict]) -> List[Dict]:
    """Tìm tất cả chunks chứa bảng markdown."""
    table_chunks = []
    
    # Pattern để nhận diện bảng markdown
    # Bảng markdown có dạng: | col1 | col2 | với dòng separator |---|---|
    table_pattern = re.compile(r'\|[^\n]+\|[\s\S]*?\|[-:]+\|', re.MULTILINE)
    
    for chunk in chunks:
        content = chunk['content']
        if table_pattern.search(content):
            table_chunks.append(chunk)
    
    return table_chunks


def display_chunk(chunk: Dict, index: int = None):
    """Hiển thị một chunk với format đẹp sử dụng Rich."""
    # Lấy thông tin metadata
    chunk_id = chunk.get('id', 'N/A')
    file_name = chunk['metadata'].get('file_name', 'N/A')
    heading_hierarchy = chunk['metadata'].get('heading_hierarchy', [])
    hierarchy_str = ' > '.join(heading_hierarchy) if heading_hierarchy else 'N/A'
    content = chunk.get('content', '')
    char_count = len(content)
    word_count = len(content.split())
    
    # Tạo tiêu đề
    if index is not None:
        title = f"📄 CHUNK #{index}"
    else:
        title = "📄 CHUNK"
    
    # Tạo phần header với thông tin metadata
    header_info = f"""🆔 [bold cyan]ID:[/bold cyan] {chunk_id}
📂 [bold cyan]File:[/bold cyan] {file_name}
🔗 [bold cyan]Heading Hierarchy:[/bold cyan] {hierarchy_str}
📊 [bold cyan]Độ dài:[/bold cyan] {char_count:,} ký tự | {word_count:,} từ"""
    
    # Tạo panel cho nội dung
    content_panel = Panel(
        content,
        title="📝 NỘI DUNG",
        title_align="left",
        border_style="green",
        padding=(1, 2),
    )
    
    # Tạo panel chính bao gồm tất cả
    main_panel = Panel(
        f"{header_info}\n\n" + "─" * 80 + f"\n\n{content}",
        title=title,
        title_align="left",
        border_style="blue",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    
    console.print(main_panel)
    console.print()


def display_chunk_pretty(chunk: Dict, index: int = None, show_full_content: bool = True):
    """Hiển thị một chunk với format đẹp và có thể tùy chọn hiển thị đầy đủ nội dung."""
    # Lấy thông tin metadata
    chunk_id = chunk.get('id', 'N/A')
    file_name = chunk['metadata'].get('file_name', 'N/A')
    heading_hierarchy = chunk['metadata'].get('heading_hierarchy', [])
    content = chunk.get('content', '')
    char_count = len(content)
    word_count = len(content.split())
    
    # Tạo bảng thông tin
    info_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    info_table.add_column("Key", style="cyan bold", width=20)
    info_table.add_column("Value", style="white")
    
    if index is not None:
        info_table.add_row("🔢 STT", str(index))
    info_table.add_row("🆔 ID", chunk_id)
    info_table.add_row("📂 File", file_name)
    info_table.add_row("📊 Độ dài", f"{char_count:,} ký tự | {word_count:,} từ")
    
    # Hiển thị heading hierarchy theo từng level
    if heading_hierarchy:
        hierarchy_text = ""
        for i, heading in enumerate(heading_hierarchy):
            prefix = "  " * i + ("└─ " if i > 0 else "")
            hierarchy_text += f"{prefix}[yellow]{heading}[/yellow]\n"
        info_table.add_row("🔗 Heading Hierarchy", hierarchy_text.strip())
    else:
        info_table.add_row("🔗 Heading Hierarchy", "[dim]N/A[/dim]")
    
    console.print(Panel(info_table, title="📋 THÔNG TIN CHUNK", border_style="cyan", box=box.ROUNDED))
    
    if show_full_content:
        # Hiển thị nội dung đầy đủ
        console.print(Panel(
            content,
            title="📝 NỘI DUNG ĐẦY ĐỦ",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        ))
    
    # Dấu phân cách giữa các chunks
    console.print("━" * 120, style="dim blue")
    console.print()


def display_chunk_summary(chunk: Dict, index: int = None):
    """Hiển thị tóm tắt một chunk."""
    hierarchy = ' > '.join(chunk['metadata'].get('heading_hierarchy', []))[:50]
    content_preview = chunk['content'][:100].replace('\n', ' ')
    
    if index is not None:
        console.print(f"  [dim]{index:4d}.[/dim] [cyan][{chunk['id']}][/cyan] [yellow]({len(chunk['content']):5d} chars)[/yellow] - {hierarchy}")
    else:
        console.print(f"  [cyan][{chunk['id']}][/cyan] [yellow]({len(chunk['content']):5d} chars)[/yellow] - {hierarchy}")


def show_shortest_chunks(chunks: List[Dict], top_n: int = 10):
    """Hiển thị N chunk ngắn nhất với đầy đủ nội dung."""
    console.print()
    console.print(Panel(
        f"[bold]🔍 TOP {top_n} CHUNKS NGẮN NHẤT[/bold]\n\n"
        "[dim]Hiển thị toàn bộ nội dung để đánh giá chất lượng dữ liệu[/dim]",
        border_style="red",
        box=box.DOUBLE,
    ))
    
    shortest = find_shortest_chunks(chunks, top_n)
    
    for i, chunk in enumerate(shortest, 1):
        display_chunk_pretty(chunk, i, show_full_content=True)
    
    # Bảng thống kê tóm tắt
    stats_table = Table(
        title="📊 THỐNG KÊ CHUNKS NGẮN NHẤT",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    stats_table.add_column("STT", justify="center", style="dim", width=6)
    stats_table.add_column("ID", style="cyan", width=40)
    stats_table.add_column("Ký tự", justify="right", style="yellow", width=10)
    stats_table.add_column("Từ", justify="right", style="green", width=10)
    stats_table.add_column("Heading", style="white", width=50)
    
    for i, chunk in enumerate(shortest, 1):
        hierarchy = chunk['metadata'].get('heading_hierarchy', [])
        hierarchy_str = hierarchy[-1] if hierarchy else "N/A"  # Chỉ lấy heading cuối
        stats_table.add_row(
            str(i),
            chunk['id'],
            f"{len(chunk['content']):,}",
            f"{len(chunk['content'].split()):,}",
            hierarchy_str[:50] + "..." if len(hierarchy_str) > 50 else hierarchy_str
        )
    
    console.print(stats_table)
    console.print()


def display_random_chunks(file_path: str, num_samples: int = 5):
    """
    Chọn và hiển thị ngẫu nhiên N chunks từ file JSONL.
    
    Args:
        file_path: Đường dẫn đến file JSONL
        num_samples: Số lượng chunks cần hiển thị ngẫu nhiên (mặc định 5)
    """
    # Load dữ liệu
    chunks = load_jsonl(file_path)
    
    # Chọn ngẫu nhiên
    if num_samples > len(chunks):
        num_samples = len(chunks)
        console.print(f"[yellow]⚠️ Chỉ có {len(chunks)} chunks, hiển thị tất cả[/yellow]")
    
    random_chunks = random.sample(chunks, num_samples)
    
    # Hiển thị header
    console.print()
    console.print(Panel(
        f"[bold]🎲 HIỂN THỊ NGẪU NHIÊN {num_samples} CHUNKS[/bold]\n\n"
        f"[dim]Tổng số chunks trong file: {len(chunks):,}[/dim]",
        border_style="magenta",
        box=box.DOUBLE,
    ))
    console.print()
    
    # Hiển thị từng chunk
    for i, chunk in enumerate(random_chunks, 1):
        display_chunk_pretty(chunk, i, show_full_content=True)
    
    # Hiển thị bảng tóm tắt
    summary_table = Table(
        title="📋 TÓM TẮT CÁC CHUNKS ĐÃ HIỂN THỊ",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    summary_table.add_column("STT", justify="center", style="dim", width=6)
    summary_table.add_column("ID", style="cyan", width=40)
    summary_table.add_column("Ký tự", justify="right", style="yellow", width=10)
    summary_table.add_column("File", style="green", width=30)
    summary_table.add_column("Heading cuối", style="white", width=40)
    
    for i, chunk in enumerate(random_chunks, 1):
        hierarchy = chunk['metadata'].get('heading_hierarchy', [])
        last_heading = hierarchy[-1] if hierarchy else "N/A"
        file_name = chunk['metadata'].get('file_name', 'N/A')
        summary_table.add_row(
            str(i),
            chunk['id'][:38] + ".." if len(chunk['id']) > 40 else chunk['id'],
            f"{len(chunk['content']):,}",
            file_name[:28] + ".." if len(file_name) > 30 else file_name,
            last_heading[:38] + ".." if len(last_heading) > 40 else last_heading
        )
    
    console.print(summary_table)
    console.print()


def show_all_tables(chunks: List[Dict], max_display: int = None):
    """Hiển thị tất cả chunks chứa bảng."""
    console.print()
    console.print(Panel(
        "[bold]📊 TẤT CẢ CHUNKS CHỨA BẢNG[/bold]",
        border_style="blue",
        box=box.DOUBLE,
    ))
    
    table_chunks = find_chunks_with_tables(chunks)
    
    console.print(f"\n[green]✅ Tìm thấy {len(table_chunks)} chunks chứa bảng[/green]\n")
    
    # Hiển thị danh sách tóm tắt
    console.print(Panel("📋 DANH SÁCH TÓM TẮT", border_style="cyan"))
    
    for i, chunk in enumerate(table_chunks[:50], 1):  # Chỉ hiện 50 đầu tiên trong tóm tắt
        display_chunk_summary(chunk, i)
    
    if len(table_chunks) > 50:
        console.print(f"[dim]... và {len(table_chunks) - 50} chunks khác[/dim]")
    
    # Hiển thị chi tiết một số chunks
    display_count = max_display if max_display else min(10, len(table_chunks))
    
    console.print()
    console.print(Panel(
        f"[bold]📝 CHI TIẾT {display_count} CHUNKS CHỨA BẢNG ĐẦU TIÊN[/bold]",
        border_style="green",
        box=box.DOUBLE,
    ))
    
    for i, chunk in enumerate(table_chunks[:display_count], 1):
        display_chunk_pretty(chunk, i, show_full_content=True)
    
    return table_chunks


def display_chunks_by_file(chunks: List[Dict], jsonl_file_path: str):
    """
    Yêu cầu người dùng nhập tên file và hiển thị tất cả chunks cho file đó.
    
    Args:
        chunks: Danh sách tất cả chunks đã load
        jsonl_file_path: Đường dẫn đến file JSONL
    """
    # Yêu cầu người dùng nhập tên file
    file_name = input("\n📂 Nhập tên file cần kiểm tra (ví dụ: nhi_khoa, duoc_thu_qg): ").strip()
    
    if not file_name:
        console.print("[red]❌ Tên file không được để trống![/red]")
        return
    
    # Lọc chunks theo tên file trong metadata
    matching_chunks = [
        c for c in chunks 
        if c.get('metadata', {}).get('file_name', '').lower() == file_name.lower()
    ]
    
    # Xử lý khi không tìm thấy
    if not matching_chunks:
        console.print(f"\n[red]❌ Không tìm thấy dữ liệu cho file: {file_name}[/red]")
        console.print("[dim]💡 Tiếp tục các tác vụ khác...[/dim]")
        return
    
    # Sắp xếp chunks theo chunk_index
    matching_chunks.sort(
        key=lambda x: x.get('metadata', {}).get('chunk_index', float('inf'))
    )
    
    # Hiển thị header
    console.print()
    console.print(Panel(
        f"[bold cyan]📄 CHUNKS CỦA FILE: {file_name}[/bold cyan]\n\n"
        f"[yellow]✅ Tổng số chunks tìm được: {len(matching_chunks)}[/yellow]",
        border_style="cyan",
        box=box.DOUBLE,
    ))
    console.print()
    
    # Hiển thị từng chunk
    for i, chunk in enumerate(matching_chunks, 1):
        # Lấy thông tin
        chunk_id = chunk.get('id', 'N/A')
        chunk_index = chunk.get('metadata', {}).get('chunk_index', 'N/A')
        heading_hierarchy = chunk.get('metadata', {}).get('heading_hierarchy', [])
        hierarchy_str = ' > '.join(heading_hierarchy) if heading_hierarchy else '[dim]N/A[/dim]'
        content = chunk.get('content', '')
        content_length = len(content)
        
        # Giới hạn hiển thị nội dung 3000 ký tự
        content_preview = content[:3000]
        if len(content) > 3000:
            content_preview += "\n\n[dim]... (nội dung tiếp tục)[/dim]"
        
        # Tạo khung hiển thị
        console.print(f"[bold blue]{'═' * 116}[/bold blue]")
        console.print(f"[bold cyan]🔹 CHUNK #{i}[/bold cyan]")
        console.print(f"  🆔 [bold]ID:[/bold] {chunk_id}")
        console.print(f"  📍 [bold]Chunk Index:[/bold] {chunk_index}")
        console.print(f"  🔗 [bold]Hierarchy:[/bold] {hierarchy_str}")
        console.print(f"  📏 [bold]Độ dài:[/bold] {content_length:,} ký tự")
        console.print(f"[blue]{'-' * 116}[/blue]")
        console.print(f"[white]{content_preview}[/white]")
        console.print()
    
    # Bảng tóm tắt
    console.print()
    console.print(Panel("[bold]📋 TÓM TẮT CÁC CHUNKS[/bold]", border_style="green"))
    
    summary_table = Table(box=box.ROUNDED, header_style="bold green", padding=(0, 1))
    summary_table.add_column("STT", justify="center", style="dim", width=6)
    summary_table.add_column("Index", justify="center", style="cyan", width=8)
    summary_table.add_column("ID", style="yellow", width=40)
    summary_table.add_column("Ký tự", justify="right", style="magenta", width=12)
    summary_table.add_column("Heading cuối", style="white", width=50)
    
    for i, chunk in enumerate(matching_chunks, 1):
        chunk_id = chunk.get('id', '')
        chunk_index = chunk.get('metadata', {}).get('chunk_index', 'N/A')
        content_length = len(chunk.get('content', ''))
        hierarchy = chunk.get('metadata', {}).get('heading_hierarchy', [])
        last_heading = hierarchy[-1] if hierarchy else "N/A"
        
        # Rút gọn ID nếu quá dài
        chunk_id_display = chunk_id[:38] + ".." if len(chunk_id) > 40 else chunk_id
        last_heading_display = last_heading[:48] + ".." if len(last_heading) > 50 else last_heading
        
        summary_table.add_row(
            str(i),
            str(chunk_index),
            chunk_id_display,
            f"{content_length:,}",
            last_heading_display
        )
    
    console.print(summary_table)
    console.print()
    console.print("[green]✅ Hoàn thành hiển thị. Quay lại menu chính...[/green]")
    console.print()


def analyze_chunk_lengths(chunks: List[Dict]):
    """Phân tích phân bố độ dài chunks."""
    lengths = [len(c['content']) for c in chunks]
    
    console.print()
    console.print(Panel(
        "[bold]📈 PHÂN TÍCH PHÂN BỐ ĐỘ DÀI CHUNKS[/bold]",
        border_style="yellow",
        box=box.DOUBLE,
    ))
    
    # Bảng thống kê tổng quan
    stats_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    stats_table.add_column("Metric", style="cyan bold")
    stats_table.add_column("Value", style="yellow")
    
    stats_table.add_row("📊 Tổng số chunks", f"{len(chunks):,}")
    stats_table.add_row("📏 Độ dài ngắn nhất", f"{min(lengths):,} ký tự")
    stats_table.add_row("📏 Độ dài dài nhất", f"{max(lengths):,} ký tự")
    stats_table.add_row("📏 Độ dài trung bình", f"{sum(lengths) / len(lengths):,.0f} ký tự")
    
    console.print(stats_table)
    
    # Phân bố theo khoảng
    ranges = [
        (0, 100, "0-100"),
        (100, 300, "100-300"),
        (300, 500, "300-500"),
        (500, 1000, "500-1000"),
        (1000, 2000, "1000-2000"),
        (2000, 3000, "2000-3000"),
        (3000, float('inf'), "3000+"),
    ]
    
    console.print()
    console.print(Panel("📊 PHÂN BỐ THEO KHOẢNG ĐỘ DÀI", border_style="green"))
    
    dist_table = Table(box=box.ROUNDED, header_style="bold green")
    dist_table.add_column("Khoảng", justify="right", style="cyan", width=12)
    dist_table.add_column("Số lượng", justify="right", style="yellow", width=10)
    dist_table.add_column("Tỷ lệ", justify="right", style="magenta", width=10)
    dist_table.add_column("Biểu đồ", style="green", width=50)
    
    for start, end, label in ranges:
        count = sum(1 for l in lengths if start <= l < end)
        percentage = count / len(lengths) * 100
        bar = "█" * int(percentage / 2) + "░" * (50 - int(percentage / 2))
        bar = bar[:50]  # Giới hạn chiều dài
        dist_table.add_row(
            label,
            f"{count:,}",
            f"{percentage:.1f}%",
            bar[:int(percentage / 2)] if percentage > 0 else ""
        )
    
    console.print(dist_table)
    console.print()


def main():
    """Hàm main."""
    
    JSONL_FILE = "/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot/data/chunks/medical_master_data6.jsonl"
    
    console.print()
    console.print(Panel(
        "[bold cyan]🔬 MEDICAL MASTER DATA ANALYZER[/bold cyan]\n\n"
        f"[dim]📁 File: {JSONL_FILE}[/dim]",
        border_style="cyan",
        box=box.DOUBLE,
    ))
    
    # Load data
    console.print("\n[yellow]⏳ Đang load dữ liệu...[/yellow]")
    chunks = load_jsonl(JSONL_FILE)
    console.print(f"[green]✅ Đã load {len(chunks):,} chunks[/green]")
    
    # Menu
    while True:
        console.print()
        menu_table = Table(
            title="📋 MENU",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_header=False,
        )
        menu_table.add_column("Option", style="cyan", width=5)
        menu_table.add_column("Description", style="white")
        
        menu_table.add_row("1", "Hiển thị chunk NGẮN NHẤT (với nội dung đầy đủ)")
        menu_table.add_row("2", "Hiển thị tất cả chunks chứa BẢNG")
        menu_table.add_row("3", "Phân tích phân bố độ dài")
        menu_table.add_row("4", "Hiển thị chunk DÀI NHẤT")
        menu_table.add_row("5", "Tìm chunk theo ID")
        menu_table.add_row("6", "[magenta]🎲 Hiển thị NGẪU NHIÊN chunks[/magenta]")
        menu_table.add_row("7", "[green]📂 Hiển thị chunks theo tên FILE[/green]")
        menu_table.add_row("0", "[red]Thoát[/red]")
        
        console.print(menu_table)
        
        choice = input("\nChọn (0-7): ").strip()
        
        if choice == "0":
            console.print("\n[cyan]👋 Tạm biệt![/cyan]")
            break
        elif choice == "1":
            try:
                n = input("Số lượng chunks ngắn nhất cần hiển thị (mặc định 10): ").strip()
                n = int(n) if n else 10
                show_shortest_chunks(chunks, n)
            except ValueError:
                show_shortest_chunks(chunks, 10)
        elif choice == "2":
            try:
                n = input("Số lượng chunks chi tiết cần hiển thị (mặc định 10, nhập 'all' để hiện tất cả): ").strip()
                if n.lower() == 'all':
                    table_chunks = show_all_tables(chunks, max_display=len(find_chunks_with_tables(chunks)))
                else:
                    n = int(n) if n else 10
                    table_chunks = show_all_tables(chunks, max_display=n)
            except ValueError:
                table_chunks = show_all_tables(chunks, max_display=10)
        elif choice == "3":
            analyze_chunk_lengths(chunks)
        elif choice == "4":
            try:
                n = input("Số lượng chunks dài nhất cần hiển thị (mặc định 10): ").strip()
                n = int(n) if n else 10
                console.print()
                console.print(Panel(
                    f"[bold]🔍 TOP {n} CHUNKS DÀI NHẤT[/bold]",
                    border_style="green",
                    box=box.DOUBLE,
                ))
                longest = find_longest_chunks(chunks, n)
                for i, chunk in enumerate(longest, 1):
                    display_chunk_pretty(chunk, i, show_full_content=True)
            except ValueError:
                pass
        elif choice == "5":
            chunk_id = input("Nhập ID chunk: ").strip()
            found = [c for c in chunks if c['id'] == chunk_id]
            if found:
                display_chunk_pretty(found[0], show_full_content=True)
            else:
                console.print(f"[red]❌ Không tìm thấy chunk với ID: {chunk_id}[/red]")
        elif choice == "6":
            try:
                n = input("Số lượng chunks ngẫu nhiên cần hiển thị (mặc định 5): ").strip()
                n = int(n) if n else 5
                display_random_chunks(JSONL_FILE, n)
            except ValueError:
                display_random_chunks(JSONL_FILE, 5)
        elif choice == "7":
            display_chunks_by_file(chunks, JSONL_FILE)
        else:
            console.print("[red]❌ Lựa chọn không hợp lệ![/red]")


if __name__ == "__main__":
    main()
