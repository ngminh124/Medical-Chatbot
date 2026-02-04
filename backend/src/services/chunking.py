"""
Document Chunking Service
Đọc và xử lý file JSONL chứa các chunks y tế
"""
import json
from typing import List, Dict, Any


class DocumentChunker:
    """Service để đọc và xử lý các chunks từ file JSONL"""
    
    def __init__(self):
        pass
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Đọc file JSONL và trả về danh sách các chunks
        
        Args:
            file_path: Đường dẫn đến file JSONL
            
        Returns:
            List các chunks với id, content và metadata
        """
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    chunks.append(chunk)
        
        print(f"📖 Đã đọc {len(chunks)} chunks từ {file_path}")
        return chunks
