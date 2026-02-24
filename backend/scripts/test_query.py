#!/usr/bin/env python3
"""
Test Query Script - Vietnamese Medical Chatbot
================================================
Script tương tác để test hệ thống RAG:
  1. Kết nối Qdrant & Embedding Service
  2. Nhận câu hỏi từ người dùng
  3. Embedding câu hỏi → Vector search → Trả về context
  4. (Tuỳ chọn) Gọi LLM để sinh câu trả lời từ context

Yêu cầu:
  - Qdrant đang chạy (docker-compose up -d)
  - Embedding service đang chạy (serving/qwen3_models/app.py)

Cách chạy:
  cd Vietnamese-Medical-Chatbot
  python -m backend.scripts.test_query
"""

import sys
import os
import textwrap
from pathlib import Path

# ── Đảm bảo project root nằm trong sys.path ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

from backend.src.configs.setup import get_backend_settings
from backend.src.services.embedding import get_embedding_service
from backend.src.core.vectorize import get_qdrant_client, search_vectors

# ── Config ──
console = Console()
settings = get_backend_settings()

# Một số câu hỏi mẫu để test nhanh
SAMPLE_QUESTIONS = [
    "Triệu chứng của sốt xuất huyết ở trẻ em là gì?",
    "Cách điều trị viêm phổi ở trẻ sơ sinh?",
    "Liều dùng paracetamol cho trẻ em như thế nào?",
    "Dấu hiệu nhận biết bệnh Kawasaki?",
    "Xử trí cấp cứu ngừng tim ở trẻ em?",
    "Chế độ dinh dưỡng cho trẻ suy dinh dưỡng?",
    "Phác đồ điều trị hen phế quản ở trẻ em?",
    "Triệu chứng và điều trị bệnh tay chân miệng?",
]


# ─────────────────────────────────────────────
#  Kiểm tra kết nối các service
# ─────────────────────────────────────────────
def check_services() -> bool:
    """Kiểm tra Qdrant và Embedding Service có sẵn sàng không."""
    all_ok = True

    # Check Qdrant
    console.print("\n[bold cyan]🔍 Kiểm tra kết nối...[/bold cyan]")
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        col_names = [c.name for c in collections]
        console.print(f"  ✅ Qdrant OK — Collections: {col_names}")

        # Kiểm tra collection mặc định có tồn tại không
        if settings.default_collection_name not in col_names:
            console.print(
                f"  ⚠️  Collection [bold]{settings.default_collection_name}[/bold] "
                f"chưa tồn tại! Hãy chạy ingest trước."
            )
            all_ok = False
        else:
            col_info = client.get_collection(settings.default_collection_name)
            console.print(
                f"  📊 Collection [bold]{settings.default_collection_name}[/bold]: "
                f"{col_info.points_count:,} points, "
                f"vector dim = {col_info.config.params.vectors.size}"
            )
    except Exception as e:
        console.print(f"  ❌ Qdrant FAIL: {e}")
        all_ok = False

    # Check Embedding Service
    try:
        emb_service = get_embedding_service()
        is_healthy = emb_service.health_check()
        if is_healthy:
            console.print(f"  ✅ Embedding Service OK — {emb_service.local_url}")
        else:
            console.print(f"  ❌ Embedding Service không phản hồi tại {emb_service.local_url}")
            all_ok = False
    except Exception as e:
        console.print(f"  ❌ Embedding Service FAIL: {e}")
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────
#  Thực hiện tìm kiếm
# ─────────────────────────────────────────────
def search_query(query: str, top_k: int = 5) -> list:
    """
    Embed câu hỏi và tìm kiếm trong Qdrant.
    Trả về list kết quả [{score, title, content, ...}]
    """
    emb_service = get_embedding_service()

    console.print(f"\n[dim]⏳ Đang tạo embedding cho câu hỏi...[/dim]")
    query_vector = emb_service.embed_query(query, use_cache=False)

    if not query_vector:
        console.print("[bold red]❌ Không thể tạo embedding cho câu hỏi![/bold red]")
        return []

    console.print(f"[dim]⏳ Đang tìm kiếm trong Qdrant (top_k={top_k})...[/dim]")
    results = search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        collection_name=settings.default_collection_name,
    )

    return results


# ─────────────────────────────────────────────
#  Hiển thị kết quả
# ─────────────────────────────────────────────
def display_results(query: str, results: list):
    """Hiển thị kết quả tìm kiếm dạng đẹp với Rich."""

    if not results:
        console.print(Panel(
            "[bold red]Không tìm thấy kết quả nào![/bold red]",
            title="Kết quả",
        ))
        return

    # Header
    console.print(Panel(
        f"[bold green]Tìm thấy {len(results)} kết quả cho:[/bold green]\n"
        f"[italic]\"{query}\"[/italic]",
        title="🔎 Kết quả tìm kiếm",
        border_style="green",
    ))

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        title = result.get("title", "Không có tiêu đề")
        content = result.get("content", "Không có nội dung")

        # Cắt nội dung nếu quá dài
        max_content_len = 800
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        # Colour score
        if score >= 0.8:
            score_color = "bold green"
        elif score >= 0.6:
            score_color = "bold yellow"
        else:
            score_color = "bold red"

        panel_content = (
            f"[bold]📄 {title}[/bold]\n"
            f"[{score_color}]Score: {score:.4f}[/{score_color}]\n"
            f"{'─' * 60}\n"
            f"{content}"
        )

        console.print(Panel(
            panel_content,
            title=f"Kết quả #{i}",
            border_style="blue",
            padding=(1, 2),
        ))


# ─────────────────────────────────────────────
#  Tạo câu trả lời từ context (nếu có LLM)
# ─────────────────────────────────────────────
def build_prompt_from_results(query: str, results: list) -> str:
    """Xây dựng prompt cho LLM từ kết quả tìm kiếm."""

    context_parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        context_parts.append(f"[Tài liệu {i}] {title}\n{content}")

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""\
    Bạn là một bác sĩ chuyên khoa nhi giàu kinh nghiệm. Hãy trả lời câu hỏi y khoa
    dựa trên các tài liệu tham khảo được cung cấp bên dưới. Trả lời bằng tiếng Việt,
    chính xác và dễ hiểu. Nếu thông tin không đủ, hãy nói rõ.

    === TÀI LIỆU THAM KHẢO ===
    {context_text}

    === CÂU HỎI ===
    {query}

    === TRẢ LỜI ===
    """)

    return prompt


def try_generate_answer(query: str, results: list):
    """
    Thử gọi LLM (nếu có) để sinh câu trả lời.
    Hỗ trợ: OpenAI-compatible API, hoặc local Ollama.
    """
    prompt = build_prompt_from_results(query, results)

    # Thử gọi Ollama (nếu có)
    try:
        import httpx
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

        console.print(f"\n[dim]⏳ Đang gọi LLM ({ollama_model}) để sinh câu trả lời...[/dim]")

        response = httpx.post(
            f"{ollama_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                },
            },
            timeout=120.0,
        )

        if response.status_code == 200:
            answer = response.json().get("response", "")
            console.print(Panel(
                answer,
                title="🤖 Câu trả lời từ LLM",
                border_style="magenta",
                padding=(1, 2),
            ))
            return True
        else:
            logger.debug(f"Ollama response: {response.status_code}")
    except Exception as e:
        logger.debug(f"Ollama not available: {e}")

    # Nếu không có LLM, hiển thị prompt để user copy
    console.print(Panel(
        "[yellow]Không tìm thấy LLM (Ollama) để sinh câu trả lời tự động.\n"
        "Bạn có thể copy prompt bên dưới và paste vào ChatGPT/Gemini/Claude...[/yellow]",
        title="💡 Gợi ý",
        border_style="yellow",
    ))

    console.print(Panel(
        prompt,
        title="📋 Prompt (copy để sử dụng với LLM)",
        border_style="dim",
        padding=(0, 1),
    ))
    return False


# ─────────────────────────────────────────────
#  Main interactive loop
# ─────────────────────────────────────────────
def show_sample_questions():
    """Hiển thị danh sách câu hỏi mẫu."""
    table = Table(
        title="📝 Câu hỏi mẫu",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="cyan", width=4)
    table.add_column("Câu hỏi", style="white")

    for i, q in enumerate(SAMPLE_QUESTIONS, 1):
        table.add_row(str(i), q)

    console.print(table)
    console.print("[dim]Gõ số (1-8) để chọn câu hỏi mẫu, hoặc nhập câu hỏi của bạn.[/dim]\n")


def main():
    console.print(Panel(
        "[bold cyan]🏥 Vietnamese Medical Chatbot — Test Query[/bold cyan]\n\n"
        f"Qdrant:     {settings.qdrant_host}:{settings.qdrant_port}\n"
        f"Collection: {settings.default_collection_name}\n"
        f"Embedding:  {settings.embedding_model_name}\n"
        f"Top-K:      {settings.top_k}",
        title="⚙️  Cấu hình",
        border_style="cyan",
    ))

    # Kiểm tra services
    if not check_services():
        console.print("\n[bold red]⛔ Một số service chưa sẵn sàng. Hãy kiểm tra lại![/bold red]")
        console.print(
            "[dim]  • Qdrant: docker-compose up -d\n"
            "  • Embedding: cd serving/qwen3_models && uvicorn app:app --port 7860[/dim]"
        )
        sys.exit(1)

    console.print("\n[bold green]✅ Tất cả services đã sẵn sàng![/bold green]\n")

    show_sample_questions()

    # Vòng lặp hỏi-đáp
    top_k = settings.top_k

    while True:
        try:
            console.print("[bold cyan]─[/bold cyan]" * 60)
            user_input = console.input("[bold cyan]❓ Nhập câu hỏi (q=thoát, s=câu mẫu, k=đổi top_k): [/bold cyan]").strip()

            if not user_input:
                continue

            # Lệnh đặc biệt
            if user_input.lower() in ("q", "quit", "exit", "thoát"):
                console.print("[bold green]👋 Tạm biệt![/bold green]")
                break

            if user_input.lower() == "s":
                show_sample_questions()
                continue

            if user_input.lower() == "k":
                new_k = console.input(f"[dim]Nhập top_k mới (hiện tại={top_k}): [/dim]").strip()
                try:
                    top_k = int(new_k)
                    console.print(f"[green]✔ Top-K đã đổi thành {top_k}[/green]")
                except ValueError:
                    console.print("[red]Giá trị không hợp lệ![/red]")
                continue

            # Nếu nhập số → chọn câu mẫu
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(SAMPLE_QUESTIONS):
                    user_input = SAMPLE_QUESTIONS[idx]
                    console.print(f"[dim]→ Đã chọn: \"{user_input}\"[/dim]")
                else:
                    console.print("[red]Số không hợp lệ![/red]")
                    continue

            # Tìm kiếm
            results = search_query(user_input, top_k=top_k)
            display_results(user_input, results)

            if results:
                # Hỏi có muốn tạo câu trả lời không
                gen = console.input(
                    "\n[dim]Bạn muốn LLM sinh câu trả lời? (y/n, mặc định=y): [/dim]"
                ).strip().lower()
                if gen != "n":
                    try_generate_answer(user_input, results)

        except KeyboardInterrupt:
            console.print("\n[bold green]👋 Tạm biệt![/bold green]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Lỗi: {e}[/bold red]")
            logger.exception("Error in query loop")


if __name__ == "__main__":
    main()
