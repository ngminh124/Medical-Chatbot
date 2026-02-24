#!/usr/bin/env python3
"""
Script chuyển đổi tài liệu Markdown Y tế thành Master Data dạng .jsonl
Đảm bảo tính chính xác về kiến thức và ngữ cảnh cho hệ thống RAG.

Features:
- Phân rã dữ liệu theo cấu trúc heading markdown
- Sửa lỗi font hệ thống (ƣ -> ư, etc.)
- Sửa lỗi mất chữ đầu (fix_medical_terms)
- Loại bỏ rác PDF (số trang, DTQGVN, ký tự lặp lại)
- Smart Hierarchy Reset (xóa hierarchy khi gặp heading cấp 1 mới)
- Split by New Subject (tách thuốc mới trong Dược thư)
- Context Injection với format [BỆNH: {Tên} | MỤC: {h1} > {h2} > {h3}]
- Kế thừa phân cấp tiêu đề (Hierarchy) xuyên suốt
- Xuất ra định dạng .jsonl chuẩn cho RAG

Version: 3.0 - Tích hợp Hierarchy Processor
"""

import os
import re
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# =============================================================================
# HIERARCHY STATE - Quản lý trạng thái phân cấp tiêu đề
# =============================================================================
MAX_HEADING_LENGTH = 200  # Giới hạn độ dài tiêu đề

@dataclass
class HierarchyState:
    """
    Lưu trữ trạng thái phân cấp tiêu đề hiện tại theo cấp markdown.
    
    Theo dõi chính xác từng cấp heading:
    - h1: # (cấp 1)
    - h2: ## (cấp 2)
    - h3: ### (cấp 3)
    - h4: #### (cấp 4)
    - h5: ##### (cấp 5)
    - h6: ###### (cấp 6)
    
    Quy tắc:
    - Khi gặp heading cấp N: cập nhật cấp N, reset các cấp > N
    - Giữ nguyên các cấp < N
    - Ví dụ: có # và ### (không có ##) → hiển thị: # > ###
    """
    h1: str = ""  # # (cấp 1 - thường là tên bệnh/thuốc)
    h2: str = ""  # ## (cấp 2)
    h3: str = ""  # ### (cấp 3)
    h4: str = ""  # #### (cấp 4)
    h5: str = ""  # ##### (cấp 5)
    h6: str = ""  # ###### (cấp 6)
    
    def update_heading(self, level: int, title: str):
        """
        Cập nhật heading theo cấp và reset các cấp thấp hơn.
        
        Args:
            level: Cấp heading (1-6, tương ứng với # đến ######)
            title: Nội dung tiêu đề
        """
        if level == 1:
            self.h1 = title
            self.h2 = ""
            self.h3 = ""
            self.h4 = ""
            self.h5 = ""
            self.h6 = ""
        elif level == 2:
            self.h2 = title
            self.h3 = ""
            self.h4 = ""
            self.h5 = ""
            self.h6 = ""
        elif level == 3:
            self.h3 = title
            self.h4 = ""
            self.h5 = ""
            self.h6 = ""
        elif level == 4:
            self.h4 = title
            self.h5 = ""
            self.h6 = ""
        elif level == 5:
            self.h5 = title
            self.h6 = ""
        elif level == 6:
            self.h6 = title
    
    def set_disease_name(self, name: str):
        """Đặt tên bệnh (h1 - cấp cao nhất)."""
        self.h1 = name
        # Reset tất cả cấp thấp hơn
        self.h2 = ""
        self.h3 = ""
        self.h4 = ""
        self.h5 = ""
        self.h6 = ""
    
    @property
    def disease_name(self) -> str:
        """Trả về tên bệnh (h1)."""
        return self.h1
    
    def reset(self):
        """Reset toàn bộ hierarchy (khi chuyển file mới)."""
        self.h1 = ""
        self.h2 = ""
        self.h3 = ""
        self.h4 = ""
        self.h5 = ""
        self.h6 = ""
    
    def get_context(self) -> str:
        """
        Tạo context string cho chunk.
        Format: [{h1} | {h2} > {h3} > {h4} > ...]
        
        Hiển thị TẤT CẢ các cấp có giá trị, không bỏ qua cấp nào.
        VD: Nếu có h1, không có h2, có h3 → hiển thị: h1 > h3
        """
        if not self.h1:
            return ""
        
        parts = [self.h1]
        
        # Thu thập tất cả các heading có giá trị (từ h2 trở đi)
        muc_parts = []
        if self.h2:
            muc_parts.append(self.h2)
        if self.h3:
            muc_parts.append(self.h3)
        if self.h4:
            muc_parts.append(self.h4)
        if self.h5:
            muc_parts.append(self.h5)
        if self.h6:
            muc_parts.append(self.h6)
        
        if muc_parts:
            parts.append(' > '.join(muc_parts))
        
        return f"[{' | '.join(parts)}]"
    
    def get_headers_list(self) -> List[str]:
        """Trả về danh sách TẤT CẢ headers hiện tại có giá trị."""
        headers = []
        if self.h1:
            headers.append(self.h1)
        if self.h2:
            headers.append(self.h2)
        if self.h3:
            headers.append(self.h3)
        if self.h4:
            headers.append(self.h4)
        if self.h5:
            headers.append(self.h5)
        if self.h6:
            headers.append(self.h6)
        return headers


# =============================================================================
# REGEX PATTERNS CHO NHẬN DIỆN TIÊU ĐỀ
# =============================================================================
# Cấp 1 (###): Số La Mã, Số thứ tự lớn, BỆNH:
PATTERN_H1_ROMAN = re.compile(r'^([IVXLCDM]+)\.?\s+(.+)$')
PATTERN_H1_NUMBER = re.compile(r'^(\d+)\.?\s+(.+)$')
PATTERN_H1_DISEASE = re.compile(r'^(BỆNH|PHẦN):\s*(.+)$', re.IGNORECASE)

# Cấp 2 (####): Số La Mã + số phụ, Số phụ, Chữ hoa
PATTERN_H2_ROMAN_SUB = re.compile(r'^([IVXLCDM]+)\.(\d+)\.?\s+(.+)$')
PATTERN_H2_DOUBLE = re.compile(r'^(\d+)\.(\d+)\.?\s+(.+)$')
PATTERN_H2_ALPHA = re.compile(r'^([ABEFGHJKNOPQRSTUWYZ])\.\s+(.+)$')

# Cấp 3 (#####): Số nhỏ, Chữ thường
PATTERN_H3_TRIPLE = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.?\s+(.+)$')
PATTERN_H3_LOWERCASE_DOT = re.compile(r'^([a-z])\.\s+(.+)$')
PATTERN_H3_LOWERCASE_PAREN = re.compile(r'^([a-z])\)\s+(.+)$')

# Valid Roman numerals
VALID_ROMAN_NUMERALS = {
    'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV'
}


# =============================================================================
# MEDICAL TERM CORRECTIONS - Sửa lỗi mất chữ đầu tiên do parse lỗi
# =============================================================================
MEDICAL_TERM_CORRECTIONS = {
    # CHỈ SỬA LỖI LẶP CHỮ ĐẦU - Không sửa lỗi "mất chữ" vì gây lặp thêm
    # Pattern: Khi có 2 chữ giống nhau liền kề (VD: CChỉ, ĐĐiều, XXử)
    
    # Dạng viết hoa đầu
    'CChỉ định': 'Chỉ định',
    'CChẩn đoán': 'Chẩn đoán', 
    'CChống chỉ định': 'Chống chỉ định',
    'CChuẩn bị': 'Chuẩn bị',
    'ĐĐiều trị': 'Điều trị',
    'ĐĐịnh nghĩa': 'Định nghĩa',
    'ĐĐại cương': 'Đại cương',
    'XXử trí': 'Xử trí',
    'XXuất huyết': 'Xuất huyết',
    'TTriệu chứng': 'Triệu chứng',
    'TTai biến': 'Tai biến',
    'TTheo dõi': 'Theo dõi',
    'TTương tác': 'Tương tác',
    'BBiến chứng': 'Biến chứng',
    'LLâm sàng': 'Lâm sàng',
    'LLiều dùng': 'Liều dùng',
    'NNguyên nhân': 'Nguyên nhân',
    'DDinh dưỡng': 'Dinh dưỡng',
    'DDụng cụ': 'Dụng cụ',
    'DDịch tễ': 'Dịch tễ',
    'KKỹ thuật': 'Kỹ thuật',
    'QQuy trình': 'Quy trình',
    'PPhòng bệnh': 'Phòng bệnh',
    'PPhản ứng': 'Phản ứng',
    'SSimh lý': 'Sinh lý',
    
    # Dạng viết HOA toàn bộ
    'CCHỈ ĐỊNH': 'CHỈ ĐỊNH',
    'CCHẨN ĐOÁN': 'CHẨN ĐOÁN',
    'CCHỐNG CHỈ ĐỊNH': 'CHỐNG CHỈ ĐỊNH',
    'CCHUẨN BỊ': 'CHUẨN BỊ',
    'ĐĐIỀU TRỊ': 'ĐIỀU TRỊ',
    'ĐĐỊNH NGHĨA': 'ĐỊNH NGHĨA',
    'ĐĐẠI CƯƠNG': 'ĐẠI CƯƠNG',
    'XXỬ TRÍ': 'XỬ TRÍ',
    'XXUẤT HUYẾT': 'XUẤT HUYẾT',
    'TTRIỆU CHỨNG': 'TRIỆU CHỨNG',
    'TTAI BIẾN': 'TAI BIẾN',
    'TTHEO DÕI': 'THEO DÕI',
    'TTƯƠNG TÁC': 'TƯƠNG TÁC',
    'BBIẾN CHỨNG': 'BIẾN CHỨNG',
    'LLÂM SÀNG': 'LÂM SÀNG',
    'LLIỀU DÙNG': 'LIỀU DÙNG',
    'NNGUYÊN NHÂN': 'NGUYÊN NHÂN',
    'DDINH DƯỠNG': 'DINH DƯỠNG',
    'DDỤNG CỤ': 'DỤNG CỤ',
    'DDỊCH TỄ': 'DỊCH TỄ',
    'KKỸ THUẬT': 'KỸ THUẬT',
    'QQUY TRÌNH': 'QUY TRÌNH',
    'PPHÒNG BỆNH': 'PHÒNG BỆNH',
    'PPHẢN ỨNG': 'PHẢN ỨNG',
    'SSINH LÝ': 'SINH LÝ',
    
    # Dạng lowercase + Uppercase (vd: xXử, cChỉ, đĐiều)
    'xXử trí': 'Xử trí',
    'cChỉ định': 'Chỉ định',
    'cChẩn đoán': 'Chẩn đoán',
    'cChuẩn bị': 'Chuẩn bị',
    'đĐiều trị': 'Điều trị',
    'đĐịnh nghĩa': 'Định nghĩa',
    'đĐại cương': 'Đại cương',
    'tTriệu chứng': 'Triệu chứng',
    'tTai biến': 'Tai biến',
    'tTheo dõi': 'Theo dõi',
    'bBiến chứng': 'Biến chứng',
    'lLâm sàng': 'Lâm sàng',
    'nNguyên nhân': 'Nguyên nhân',
    'dDinh dưỡng': 'Dinh dưỡng',
    'dDụng cụ': 'Dụng cụ',
    'kKỹ thuật': 'Kỹ thuật',
    'qQuy trình': 'Quy trình',
    'pPhòng bệnh': 'Phòng bệnh',
}


# =============================================================================
# TOP-LEVEL KEYWORDS - Các từ khóa quan trọng sẽ được ép về Heading cấp 1
# =============================================================================
TOP_LEVEL_KEYWORDS = [
    'ĐẠI CƯƠNG',
    'ĐỊNH NGHĨA',
    'CHỈ ĐỊNH', 
    'CHỐNG CHỈ ĐỊNH',
    'CHUẨN BỊ',
    'QUY TRÌNH',
    'KỸ THUẬT',
    'TAI BIẾN',
    'BIẾN CHỨNG',
    'THEO DÕI',
    'XỬ TRÍ',
    'ĐIỀU TRỊ',
    'CHẨN ĐOÁN',
    'TIÊN LƯỢNG',
    'PHÒNG BỆNH',
    'TÀI LIỆU THAM KHẢO',
    'NGUYÊN NHÂN',
    'TRIỆU CHỨNG',
    'CẬN LÂM SÀNG',
    'LÂM SÀNG',
    'DỊCH TỄ HỌC',
    'SINH LÝ BỆNH',
    'CƠ CHẾ BỆNH SINH',
    'DƯỢC LÝ',
    'DƯỢC ĐỘNG HỌC',
    'THẬN TRỌNG',
    'TƯƠNG TÁC THUỐC',
    'LIỀU DÙNG',
    'TÁC DỤNG PHỤ',
    'BẢO QUẢN',
]


# =============================================================================
# FONT CORRECTIONS DICTIONARY
# =============================================================================
FONT_CORRECTIONS = {
    # Lowercase
    'đƣợc': 'được',
    'tƣơi': 'tươi', 
    'thƣờng': 'thường',
    'chƣơng': 'chương',
    'đƣờng': 'đường',
    'hƣớng': 'hướng',
    'trƣớc': 'trước',
    'dƣới': 'dưới',
    'ngƣời': 'người',
    'tƣơng': 'tương',
    'nƣớc': 'nước',
    'trƣờng': 'trường',
    'bƣớc': 'bước',
    'rƣợu': 'rượu',
    'lƣu': 'lưu',
    'lƣợng': 'lượng',
    'dƣơng': 'dương',
    'sƣởi': 'sưởi',
    'tƣởng': 'tưởng',
    'hƣởng': 'hưởng',
    'nƣơng': 'nương',
    'mƣời': 'mười',
    'tƣới': 'tưới',
    'cƣờng': 'cường',
    'xƣơng': 'xương',
    'sƣơn': 'sườn',
    'vƣợt': 'vượt',
    'khƣớc': 'khước',
    'chƣa': 'chưa',
    'mƣa': 'mưa',
    'thƣa': 'thưa',
    'xƣa': 'xưa',
    'tƣ': 'tư',
    'sƣ': 'sư',
    'cƣ': 'cư',
    'thƣ': 'thư',
    'nhƣ': 'như',
    'dƣ': 'dư',
    'ƣ': 'ư',
    
    # Thêm các pattern phổ biến còn thiếu
    'nhƣng': 'nhưng',
    'sƣng': 'sưng',
    'mƣớp': 'mướp',
    'bƣởi': 'bưởi',
    'gƣơng': 'gương',
    'vƣờn': 'vườn',
    'hƣu': 'hưu',
    'cƣỡi': 'cưỡi',
    'tƣơi': 'tươi',
    'mƣợn': 'mượn',
    'lƣỡi': 'lưỡi',
    'sƣơng': 'sương',
    
    # Uppercase
    'CHƢƠNG': 'CHƯƠNG',
    'ĐƢỜNG': 'ĐƯỜNG',
    'NGƢỜI': 'NGƯỜI',
    'THƢỜNG': 'THƯỜNG',
    'TRƢỚC': 'TRƯỚC',
    'HƢỚNG': 'HƯỚNG',
    'NƢỚC': 'NƯỚC',
    'TRƢỜNG': 'TRƯỜNG',
    'BƢỚC': 'BƯỚC',
    'DƢỚI': 'DƯỚI',
    'TƢƠNG': 'TƯƠNG',
    'CƢỜNG': 'CƯỜNG',
    'XƢƠNG': 'XƯƠNG',
    'LƢỢNG': 'LƯỢNG',
    'NHƢ': 'NHƯ',
    'DƢ': 'DƯ',
    'TƢ': 'TƯ',
    'Ƣ': 'Ư',
    
    # Title case
    'Đƣợc': 'Được',
    'Ngƣời': 'Người',
    'Thƣờng': 'Thường',
    'Trƣớc': 'Trước',
    'Hƣớng': 'Hướng',
    'Nƣớc': 'Nước',
    'Chƣơng': 'Chương',
    'Đƣờng': 'Đường',
    'Trƣờng': 'Trường',
    'Bƣớc': 'Bước',
    'Dƣới': 'Dưới',
    'Tƣơng': 'Tương',
    'Lƣu': 'Lưu',
    'Lƣợng': 'Lượng',
}


# =============================================================================
# PDF GARBAGE PATTERNS - Loại bỏ nhiễu Header/Footer (Optimized)
# =============================================================================
PDF_GARBAGE_PATTERNS = [
    # Số trang đơn lẻ (1-4 chữ số)
    r'^\s*\d{1,4}\s*$',
    
    # Page markers
    r'^\s*[-–—]\s*\d+\s*[-–—]\s*$',
    r'^\s*Trang\s+\d+\s*$',
    r'^\s*Page\s+\d+\s*$',
    
    # Image placeholders
    r'^\s*!\[\]\([^)]*\)\s*$',
    r'^\s*\[image\]\s*$',
    
    # Empty markdown elements
    r'^\s*\*{1,2}\s*\*{1,2}\s*$',
    r'^\s*_{1,2}\s*_{1,2}\s*$',
    
    # Repeated dashes/equals (header/footer decorations)
    r'^\s*[-=]{5,}\s*$',
]


# =============================================================================
# DRUG NAME DETECTION PATTERNS - Phát hiện tên thuốc mới trong Dược thư
# =============================================================================
# Pattern: Dòng viết hoa toàn bộ (tên thuốc) thường có dạng:
# # STREPTOKINASE hoặc # STREPTokinase
DRUG_NAME_PATTERN = re.compile(
    r'^#\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)\s*$'
)

# Pattern phát hiện dòng viết hoa toàn bộ (không có #)
ALL_CAPS_LINE_PATTERN = re.compile(
    r'^([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ\s\-]{2,})$'
)

# ATC Code pattern (ví dụ: B01AD01, C07AB02)
ATC_CODE_PATTERN = re.compile(
    r'\b([A-Z]\d{2}[A-Z]{2}\d{2})\b'
)


# =============================================================================
# HEADING DETECTION FUNCTIONS - Nhận diện cấp độ tiêu đề
# =============================================================================

def extract_clean_content(line: str) -> str:
    """Trích xuất nội dung sạch từ một dòng (loại bỏ # và **)."""
    cleaned = re.sub(r'^#+\s*', '', line.strip())
    cleaned = re.sub(r'^\*{1,2}\s*', '', cleaned)
    cleaned = re.sub(r'\s*\*{1,2}$', '', cleaned)
    return cleaned.strip()


def clean_heading_title(title: str) -> str:
    """Làm sạch tiêu đề: loại bỏ **, *, dấu : cuối."""
    title = re.sub(r'^\*{1,2}\s*', '', title)
    title = re.sub(r'\s*\*{1,2}$', '', title)
    title = title.rstrip(':').strip()
    return title


def is_valid_heading_length(text: str) -> bool:
    """Kiểm tra độ dài có hợp lệ cho tiêu đề không."""
    return len(text.strip()) <= MAX_HEADING_LENGTH


def detect_heading_level(line: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Nhận diện cấp độ tiêu đề của một dòng.
    
    Returns:
        Tuple (level, prefix, title):
        - level: 1, 2, 3 hoặc None nếu không phải tiêu đề
        - prefix: Phần đầu tiêu đề (số thứ tự, chữ cái...)
        - title: Nội dung tiêu đề
    """
    # Làm sạch dòng: loại bỏ # và ** ở đầu
    cleaned = extract_clean_content(line)
    
    # Kiểm tra độ dài hợp lệ
    if not cleaned or not is_valid_heading_length(cleaned):
        return None, None, None
    
    # ========== CẤP 3 (#####) - KIỂM TRA TRƯỚC ==========
    # (Pattern cụ thể hơn nên kiểm tra trước)
    
    # 1. Số thứ tự nhỏ (1.1.1., 2.1.1., ...)
    match = PATTERN_H3_TRIPLE.match(cleaned)
    if match:
        prefix = f"{match.group(1)}.{match.group(2)}.{match.group(3)}."
        title = clean_heading_title(match.group(4))
        return 3, prefix, title
    
    # 2. Chữ cái viết thường dạng a., b., c.
    match = PATTERN_H3_LOWERCASE_DOT.match(cleaned)
    if match:
        prefix = f"{match.group(1)}."
        title = clean_heading_title(match.group(2))
        return 3, prefix, title
    
    # 3. Chữ cái viết thường dạng a), b), c)
    match = PATTERN_H3_LOWERCASE_PAREN.match(cleaned)
    if match:
        prefix = f"{match.group(1)})"
        title = clean_heading_title(match.group(2))
        return 3, prefix, title
    
    # ========== CẤP 2 (####) ==========
    
    # 1. Số La Mã + số phụ (II.1., II.2., III.1., ...)
    match = PATTERN_H2_ROMAN_SUB.match(cleaned)
    if match:
        prefix = f"{match.group(1)}.{match.group(2)}."
        title = clean_heading_title(match.group(3))
        return 2, prefix, title
    
    # 2. Số thứ tự phụ (1.1., 2.1., 3.1., ...)
    match = PATTERN_H2_DOUBLE.match(cleaned)
    if match:
        prefix = f"{match.group(1)}.{match.group(2)}."
        title = clean_heading_title(match.group(3))
        return 2, prefix, title
    
    # 3. Chữ cái viết hoa (A., B., ...) - loại trừ I, V, X, L, C, D, M
    match = PATTERN_H2_ALPHA.match(cleaned)
    if match:
        prefix = f"{match.group(1)}."
        title = clean_heading_title(match.group(2))
        return 2, prefix, title
    
    # ========== CẤP 1 (###) ==========
    
    # 1. Số La Mã (I., II., III., ...)
    match = PATTERN_H1_ROMAN.match(cleaned)
    if match:
        roman_num = match.group(1)
        # Đảm bảo đây là số La Mã hợp lệ (không phải từ ngẫu nhiên)
        if roman_num.upper() in VALID_ROMAN_NUMERALS:
            prefix = f"{roman_num}."
            title = clean_heading_title(match.group(2))
            return 1, prefix, title
    
    # 2. Số thứ tự lớn (1., 2., 3., ...)
    match = PATTERN_H1_NUMBER.match(cleaned)
    if match:
        rest = match.group(2).strip()
        # Đảm bảo không phải cấp 2 (1.1) hoặc cấp 3 (1.1.1)
        if not re.match(r'^\d+\.', rest):
            prefix = f"{match.group(1)}."
            title = clean_heading_title(rest)
            return 1, prefix, title
    
    # 3. Tiêu đề bắt đầu bằng BỆNH: hoặc PHẦN:
    match = PATTERN_H1_DISEASE.match(cleaned)
    if match:
        prefix = f"{match.group(1).upper()}:"
        title = clean_heading_title(match.group(2))
        return 1, prefix, title
    
    return None, None, None


@dataclass
class MasterChunk:
    """Đại diện cho một chunk trong Master Data."""
    id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def fix_medical_terms(text: str) -> str:
    """
    Hàm sửa triệt để lỗi mất chữ trong thuật ngữ y khoa.
    Optimized version - chỉ áp dụng các corrections cần thiết.
    
    Args:
        text: Văn bản cần sửa
    
    Returns:
        Văn bản đã được sửa lỗi
    """
    if not text:
        return text
    
    result = text
    
    # 1. Sửa lỗi lặp chữ đầu bằng string replacement trực tiếp (nhanh và chính xác)
    # Các pattern lỗi phổ biến: xX, cC, dD, đĐ, tT, bB, nN, lL, kK, qQ, pP
    duplicate_fixes = [
        # lowercase + Uppercase patterns
        ('xXử', 'Xử'),
        ('cChỉ', 'Chỉ'),
        ('cChẩn', 'Chẩn'),
        ('cChống', 'Chống'),
        ('cChuẩn', 'Chuẩn'),
        ('dDinh', 'Dinh'),
        ('dDụng', 'Dụng'),
        ('dDịch', 'Dịch'),
        ('đĐiều', 'Điều'),
        ('đĐịnh', 'Định'),
        ('đĐại', 'Đại'),
        ('tTriệu', 'Triệu'),
        ('tTai', 'Tai'),
        ('tTheo', 'Theo'),
        ('tTương', 'Tương'),
        ('bBiến', 'Biến'),
        ('bBệnh', 'Bệnh'),
        ('nNguyên', 'Nguyên'),
        ('nNhiễm', 'Nhiễm'),
        ('lLâm', 'Lâm'),
        ('lLiều', 'Liều'),
        ('kKỹ', 'Kỹ'),
        ('qQuy', 'Quy'),
        ('qQuá', 'Quá'),
        ('pPhòng', 'Phòng'),
        ('pPhản', 'Phản'),
        ('sSimh', 'Sinh'),
        ('oOuá', 'Quá'),
        ('iIa', 'Ia'),
        ('cChuẩn', 'Chuẩn'),  # Thêm pattern cChuẩn
        ('nNguyên', 'Nguyên'),
        ('lLâm', 'Lâm'),
        ('kKỹ', 'Kỹ'),
        ('tTai', 'Tai'),
        ('bBiến', 'Biến'),
        # Uppercase + Uppercase patterns 
        ('XXử', 'Xử'),
        ('XXỬ', 'XỬ'),
        ('XXUẤT', 'XUẤT'),  # Thêm pattern XXUẤT
        ('CChỉ', 'Chỉ'),
        ('CCHỈ', 'CHỈ'),
        ('CChẩn', 'Chẩn'),
        ('CCHẨN', 'CHẨN'),
        ('CChống', 'Chống'),
        ('CCHỐNG', 'CHỐNG'),
        ('CChuẩn', 'Chuẩn'),
        ('CCHUẨN', 'CHUẨN'),
        ('DDinh', 'Dinh'),
        ('DDINH', 'DINH'),
        ('DDụng', 'Dụng'),
        ('DDỤNG', 'DỤNG'),
        ('DDịch', 'Dịch'),
        ('DDỊCH', 'DỊCH'),
        ('ĐĐiều', 'Điều'),
        ('ĐĐIỀU', 'ĐIỀU'),
        ('ĐĐịnh', 'Định'),
        ('ĐĐỊNH', 'ĐỊNH'),
        ('ĐĐại', 'Đại'),
        ('ĐĐẠI', 'ĐẠI'),
        ('TTriệu', 'Triệu'),
        ('TTRIỆU', 'TRIỆU'),
        ('TTai', 'Tai'),
        ('TTAI', 'TAI'),
        ('TTheo', 'Theo'),
        ('TTHEO', 'THEO'),
        ('TTương', 'Tương'),
        ('TTƯƠNG', 'TƯƠNG'),
        ('BBiến', 'Biến'),
        ('BBIẾN', 'BIẾN'),
        ('BBệnh', 'Bệnh'),
        ('BBỆNH', 'BỆNH'),
        ('NNguyên', 'Nguyên'),
        ('NNGUYÊN', 'NGUYÊN'),
        ('NNhiễm', 'Nhiễm'),
        ('NNHIỄM', 'NHIỄM'),
        ('LLâm', 'Lâm'),
        ('LLÂM', 'LÂM'),
        ('LLiều', 'Liều'),
        ('LLIỀU', 'LIỀU'),
        ('KKỹ', 'Kỹ'),
        ('KKỸ', 'KỸ'),
        ('QQuy', 'Quy'),
        ('QQUY', 'QUY'),
        ('QQuá', 'Quá'),
        ('QQUÁ', 'QUÁ'),
        ('PPhòng', 'Phòng'),
        ('PPHÒNG', 'PHÒNG'),
        ('PPhản', 'Phản'),
        ('PPHẢN', 'PHẢN'),
        ('SSimh', 'Sinh'),
        ('SSINH', 'SINH'),
        # Các lỗi đặc biệt khác
        ('Ouá liều', 'Quá liều'),
        ('OUÁ LIỀU', 'QUÁ LIỀU'),
        ('ia chảy', 'Ia chảy'),  # "i" -> "I" nếu bị lỗi
    ]
    
    for wrong, correct in duplicate_fixes:
        if wrong in result:
            result = result.replace(wrong, correct)
    
    # 2. Áp dụng dictionary corrections (chỉ những từ có trong text)
    for wrong, correct in MEDICAL_TERM_CORRECTIONS.items():
        if wrong in result:  # Fast check before replace
            result = result.replace(wrong, correct)
    
    return result


def is_new_drug_subject(line: str) -> Optional[str]:
    """
    Kiểm tra xem dòng có phải là tên thuốc mới trong Dược thư không.
    
    Tên thuốc mới thường là:
    - Dòng heading với tên viết hoa toàn bộ hoặc phần lớn viết hoa
    - Ví dụ: # STREPTOKINASE, # Paracetamol, # ACARBOSE
    
    Args:
        line: Dòng văn bản cần kiểm tra
    
    Returns:
        Tên thuốc nếu là thuốc mới, None nếu không phải
    """
    stripped = line.strip()
    
    # Kiểm tra pattern heading với tên thuốc
    # # TÊN_THUỐC (viết hoa hoặc title case)
    match = re.match(r'^#\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s\-\(\)]+)$', stripped)
    if match:
        drug_name = match.group(1).strip()
        
        # Kiểm tra: Ít nhất 50% chữ cái là viết hoa hoặc tên ngắn (< 20 ký tự)
        upper_count = sum(1 for c in drug_name if c.isupper())
        alpha_count = sum(1 for c in drug_name if c.isalpha())
        
        if alpha_count > 0:
            upper_ratio = upper_count / alpha_count
            
            # Là thuốc mới nếu: > 50% viết hoa HOẶC là tên ngắn
            if upper_ratio > 0.5 or len(drug_name) < 20:
                # Loại trừ các từ khóa phổ biến (không phải tên thuốc)
                keywords_to_exclude = [
                    'ĐẠI CƯƠNG', 'CHỈ ĐỊNH', 'CHỐNG CHỈ ĐỊNH', 'LIỀU DÙNG',
                    'TÁC DỤNG', 'THẬN TRỌNG', 'TƯƠNG TÁC', 'BẢO QUẢN',
                    'DƯỢC LÝ', 'DƯỢC ĐỘNG', 'QUÁ LIỀU', 'ĐIỀU TRỊ',
                ]
                if not any(kw in drug_name.upper() for kw in keywords_to_exclude):
                    return drug_name
    
    return None


def extract_atc_code(text: str) -> Optional[str]:
    """
    Trích xuất mã ATC từ văn bản.
    
    Mã ATC có định dạng: 1 chữ cái + 2 số + 2 chữ cái + 2 số
    Ví dụ: B01AD01, C07AB02, N02BE01
    
    Args:
        text: Văn bản cần tìm mã ATC
    
    Returns:
        Mã ATC đầu tiên tìm thấy hoặc None
    """
    match = ATC_CODE_PATTERN.search(text)
    return match.group(1) if match else None


class MarkdownHeaderTextSplitter:
    """
    Splitter chia văn bản markdown theo cấu trúc heading.
    Giữ nguyên thứ bậc tiêu đề cho mỗi đoạn.
    
    Cải tiến v2:
    - Smart heading detection: Tự động nhận diện và ép các heading chứa từ khóa
      quan trọng (ĐẠI CƯƠNG, CHỈ ĐỊNH, etc.) về cấp 1
    - Chuẩn hóa tiêu đề: Loại bỏ số la mã, ký tự thừa
    - Smart Hierarchy Reset: Xóa sạch hierarchy khi gặp heading cấp 1 mới
    - Split by New Subject: Tách thuốc mới trong Dược thư
    """
    
    def __init__(self, headers_to_split_on: List[Tuple[str, str]] = None):
        """
        Args:
            headers_to_split_on: List of tuples (header_marker, header_name)
                Example: [("#", "H1"), ("##", "H2"), ("###", "H3")]
        """
        if headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "Heading1"),
                ("##", "Heading2"),
                ("###", "Heading3"),
                ("####", "Heading4"),
                ("#####", "Heading5"),
                ("######", "Heading6"),
            ]
        else:
            self.headers_to_split_on = headers_to_split_on
        
        # Sort by length descending to match longer headers first
        self.headers_to_split_on = sorted(
            self.headers_to_split_on, 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        # Biên dịch regex để nhận diện số la mã và số thường ở đầu tiêu đề
        self.numbering_pattern = re.compile(
            r'^\s*'
            r'(?:'
            r'[IVXLCDM]+\.?\s*'
            r'|[0-9]+\.?\s*'
            r'|[a-zA-Z]\.\s*'
            r'|[-–•]\s*'
            r')*'
            r'[:\-–]?\s*',
            re.IGNORECASE
        )
    
    def _normalize_header_text(self, header_text: str) -> str:
        """
        Chuẩn hóa tiêu đề: loại bỏ số la mã, số thường, ký tự thừa.
        Ví dụ:
            "III. ĐẠI CƯƠNG" -> "ĐẠI CƯƠNG"
            "1.2.3 Chỉ định" -> "Chỉ định"
            "**CHỈ ĐỊNH:**" -> "CHỈ ĐỊNH"
            "hỉ định" -> "Chỉ định"  (sửa lỗi mất chữ đầu)
        """
        # Loại bỏ bold markers
        text = re.sub(r'^\*{1,2}|\*{1,2}$', '', header_text).strip()
        
        # Loại bỏ dấu : ở cuối
        text = re.sub(r'[:\-–]+\s*$', '', text).strip()
        
        # Loại bỏ số la mã, số thường ở đầu
        text = self._safe_remove_numbering(text).strip()
        
        # ÁP DỤNG SỬA LỖI MẤT CHỮ ĐẦU
        text = self._fix_heading_typo(text)
        
        return text
    
    def _safe_remove_numbering(self, text: str) -> str:
        """
        Loại bỏ số thứ tự đầu dòng một cách an toàn, không xóa ký tự đầu tiên của từ.
        Xử lý: I., II., 1., 1.1, a., -, • 
        """
        patterns = [
            r'^\s*([IVXLCDM]+)\.\s+',           # Roman numerals: I., II., III.
            r'^\s*(\d+(?:\.\d+)*)\.?\s+',       # Numbers: 1., 1.1., 1.2.3
            r'^\s*([a-zA-Z])\.\s+',              # Letters: a., b., A.
            r'^\s*[-–•]\s+',                     # Bullets: -, –, •
        ]
        
        result = text
        for pattern in patterns:
            match = re.match(pattern, result)
            if match:
                result = re.sub(pattern, '', result, count=1)
                break
        
        return result
    
    def _fix_heading_typo(self, text: str) -> str:
        """
        Sửa lỗi mất chữ đầu tiên trong tiêu đề.
        Ví dụ: 'hỉ định' -> 'Chỉ định'
        """
        for wrong, correct in MEDICAL_TERM_CORRECTIONS.items():
            if text.lower() == wrong.lower():
                return correct
            if text.lower().startswith(wrong.lower() + ' '):
                return correct + text[len(wrong):]
            if text.lower().startswith(wrong.lower()):
                remaining = text[len(wrong):]
                if not remaining or remaining[0] in ' :.,;!':
                    return correct + remaining
        
        return text
    
    def _is_top_level_keyword(self, header_text: str) -> bool:
        """
        Kiểm tra xem tiêu đề có chứa từ khóa top-level không.
        """
        normalized = self._normalize_header_text(header_text).upper()
        
        for keyword in TOP_LEVEL_KEYWORDS:
            if normalized == keyword or normalized.startswith(keyword):
                return True
        
        return False
    
    def _has_roman_numeral_prefix(self, text: str) -> bool:
        """
        Kiểm tra xem text có bắt đầu bằng số La Mã không.
        """
        roman_pattern = r'^\s*\*{0,2}\s*([IVXLCDM]+)\.\s+'
        return bool(re.match(roman_pattern, text, re.IGNORECASE))
    
    def _parse_header(self, line: str) -> Optional[Tuple[int, str, str]]:
        """
        Parse a line to check if it's a header.
        Returns: (level, header_name, header_text) or None
        """
        stripped = line.strip()
        
        for header_marker, header_name in self.headers_to_split_on:
            if stripped.startswith(header_marker + " "):
                remaining = stripped[len(header_marker):]
                if remaining and remaining[0] == '#':
                    continue
                
                original_level = len(header_marker)
                raw_header_text = remaining.strip()
                
                has_roman = self._has_roman_numeral_prefix(raw_header_text)
                header_text = self._normalize_header_text(raw_header_text)
                
                if has_roman:
                    level = 1
                    header_name = "Heading1"
                elif self._is_top_level_keyword(header_text):
                    level = 1
                    header_name = "Heading1"
                else:
                    level = original_level
                
                return (level, header_name, header_text)
        
        return None
    
    def _is_h1_disease_heading(self, line: str) -> Optional[str]:
        """
        Kiểm tra xem dòng có phải là Heading cấp 1 cho bệnh mới không.
        
        Pattern: # TÊN BỆNH VIẾT HOA TOÀN BỘ
        Ví dụ: # BỆNH VIÊM NÃO DO VIRUS HERPES SIMPLEX
        
        Returns:
            Tên bệnh nếu là H1 bệnh mới, None nếu không phải
        """
        stripped = line.strip()
        
        # Kiểm tra pattern: # TÊN VIẾT HOA
        if stripped.startswith('# '):
            heading_text = stripped[2:].strip()
            # Loại bỏ dấu * nếu có
            heading_text = re.sub(r'^\*+|\*+$', '', heading_text).strip()
            
            # Kiểm tra xem có phải viết hoa toàn bộ không
            upper_count = sum(1 for c in heading_text if c.isupper())
            alpha_count = sum(1 for c in heading_text if c.isalpha())
            
            if alpha_count > 0:
                upper_ratio = upper_count / alpha_count
                
                # Là bệnh mới nếu > 70% viết hoa VÀ không phải từ khóa thông thường
                if upper_ratio > 0.7 and len(heading_text) > 10:
                    # Loại trừ các từ khóa top-level (ĐẠI CƯƠNG, CHỈ ĐỊNH, etc.)
                    is_top_level_kw = any(
                        heading_text.upper() == kw or heading_text.upper().startswith(kw + ' ')
                        for kw in TOP_LEVEL_KEYWORDS
                    )
                    if not is_top_level_kw:
                        return heading_text
        
        return None
    
    def _is_reference_section(self, header_text: str) -> bool:
        """
        Kiểm tra xem tiêu đề có phải là phần Tài liệu tham khảo không.
        """
        normalized = header_text.upper().strip()
        reference_keywords = [
            'TÀI LIỆU THAM KHẢO',
            'THAM KHẢO',
            'TÀI LIỆU',
            'REFERENCES',
            'BIBLIOGRAPHY',
        ]
        return any(kw in normalized for kw in reference_keywords)
    
    def split_text(self, text: str, is_drug_document: bool = False) -> List[Dict]:
        """
        Split text by markdown headers với Smart Hierarchy Reset.
        
        Args:
            text: Văn bản markdown
            is_drug_document: True nếu là file Dược thư (cần split theo tên thuốc)
        
        Returns:
            List of dicts with 'content', 'metadata', và 'subject_name' (tên thuốc/bệnh)
        """
        lines = text.split('\n')
        chunks = []
        
        # Stack to track current header hierarchy: [(level, text), ...]
        header_stack = []
        current_content_lines = []
        current_subject_name = None  # Tên thuốc/bệnh hiện tại
        skip_reference_section = False  # Flag để bỏ qua phần tài liệu tham khảo
        
        for line in lines:
            # ======= KIỂM TRA BỆNH MỚI (H1 VIẾT HOA TOÀN BỘ) =======
            if not is_drug_document:
                new_disease = self._is_h1_disease_heading(line)
                if new_disease:
                    # Lưu chunk cũ trước khi bắt đầu bệnh mới
                    if current_content_lines and not skip_reference_section:
                        content = '\n'.join(current_content_lines).strip()
                        if content:
                            chunks.append({
                                'content': content,
                                'metadata': {
                                    'headers': [h[1] for h in header_stack],
                                    'header_levels': [h[0] for h in header_stack]
                                },
                                'subject_name': current_subject_name
                            })
                        current_content_lines = []
                    
                    # XÓA SẠCH HIERARCHY & RESET SUBJECT - Bắt đầu bệnh mới
                    header_stack = []
                    current_subject_name = new_disease  # CẬP NHẬT TÊN BỆNH MỚI
                    skip_reference_section = False  # Reset flag tài liệu tham khảo
                    
                    # Push tên bệnh như heading cấp 1
                    header_stack.append((1, new_disease))
                    current_content_lines.append(line)
                    continue
            
            # ======= KIỂM TRA THUỐC MỚI (SPLIT BY NEW SUBJECT) =======
            if is_drug_document:
                new_drug = is_new_drug_subject(line)
                if new_drug:
                    # Lưu chunk cũ trước khi bắt đầu thuốc mới
                    if current_content_lines and not skip_reference_section:
                        content = '\n'.join(current_content_lines).strip()
                        if content:
                            chunks.append({
                                'content': content,
                                'metadata': {
                                    'headers': [h[1] for h in header_stack],
                                    'header_levels': [h[0] for h in header_stack]
                                },
                                'subject_name': current_subject_name
                            })
                        current_content_lines = []
                    
                    # XÓA SẠCH HIERARCHY - Bắt đầu thuốc mới
                    header_stack = []
                    current_subject_name = new_drug
                    skip_reference_section = False  # Reset flag
                    
                    # Push tên thuốc như heading cấp 1
                    header_stack.append((1, new_drug))
                    current_content_lines.append(line)
                    continue
            
            # ======= PARSE HEADER =======
            header_info = self._parse_header(line)
            
            if header_info:
                level, header_name, header_text = header_info
                
                # ======= KIỂM TRA TÀI LIỆU THAM KHẢO =======
                if self._is_reference_section(header_text):
                    # Lưu chunk hiện tại trước khi bỏ qua phần tài liệu tham khảo
                    if current_content_lines and not skip_reference_section:
                        content = '\n'.join(current_content_lines).strip()
                        if content:
                            chunks.append({
                                'content': content,
                                'metadata': {
                                    'headers': [h[1] for h in header_stack],
                                    'header_levels': [h[0] for h in header_stack]
                                },
                                'subject_name': current_subject_name
                            })
                        current_content_lines = []
                    
                    # Bật flag để bỏ qua nội dung tiếp theo
                    skip_reference_section = True
                    continue  # Không thêm heading này vào hierarchy
                
                # Nếu đang trong phần tài liệu tham khảo, kiểm tra xem có phải heading cấp 1 mới không
                # để thoát khỏi phần tham khảo
                if skip_reference_section:
                    if level == 1:
                        # Gặp heading cấp 1 mới -> thoát khỏi phần tài liệu tham khảo
                        skip_reference_section = False
                    else:
                        # Vẫn trong phần tài liệu tham khảo -> bỏ qua
                        continue
                
                # Lưu chunk hiện tại nếu có nội dung
                if current_content_lines:
                    content = '\n'.join(current_content_lines).strip()
                    if content:
                        chunks.append({
                            'content': content,
                            'metadata': {
                                'headers': [h[1] for h in header_stack],
                                'header_levels': [h[0] for h in header_stack]
                            },
                            'subject_name': current_subject_name
                        })
                    current_content_lines = []
                
                # ======= SMART HIERARCHY RESET =======
                # Khi gặp heading cấp 1 mới -> XÓA SẠCH HIERARCHY VÀ CẬP NHẬT SUBJECT
                if level == 1:
                    # Kiểm tra xem có phải bắt đầu bệnh/thuốc mới không
                    # (không phải từ khóa top-level như ĐẠI CƯƠNG, CHỈ ĐỊNH...)
                    is_new_subject = not any(
                        header_text.upper() == kw or header_text.upper().startswith(kw + ' ')
                        for kw in TOP_LEVEL_KEYWORDS
                    )
                    
                    if is_new_subject and len(header_text) > 10:
                        # Đây là bệnh/thuốc mới -> CẬP NHẬT current_subject_name
                        current_subject_name = header_text
                    elif not current_subject_name:
                        # Chưa có subject -> dùng header này
                        current_subject_name = header_text
                    
                    # XÓA SẠCH hierarchy cũ
                    header_stack = []
                else:
                    # Pop tất cả headers có level >= current level
                    while header_stack and header_stack[-1][0] >= level:
                        header_stack.pop()
                
                # Push current header
                header_stack.append((level, header_text))
                current_content_lines.append(line)
            else:
                # Nếu đang trong phần tài liệu tham khảo -> bỏ qua dòng này
                if skip_reference_section:
                    continue
                current_content_lines.append(line)
        
        # Chunk cuối cùng (chỉ thêm nếu không phải phần tài liệu tham khảo)
        if current_content_lines and not skip_reference_section:
            content = '\n'.join(current_content_lines).strip()
            if content:
                chunks.append({
                    'content': content,
                    'metadata': {
                        'headers': [h[1] for h in header_stack],
                        'header_levels': [h[0] for h in header_stack]
                    },
                    'subject_name': current_subject_name
                })
        
        return chunks


class MedicalMarkdownConverter:
    """
    Converter chuyển đổi tài liệu Markdown Y tế thành Master Data JSONL.
    
    Version 2.0 với các cải tiến:
    - Fix medical terms triệt để
    - Noise removal (DTQGVN, số trang, ký tự lặp)
    - Smart hierarchy reset
    - Split by new subject (Dược thư)
    - Context injection với format chuẩn
    """
    
    # Danh sách folder chứa Dược thư (cần split theo tên thuốc)
    DRUG_DOCUMENT_FOLDERS = [
        'duoc_thu_qg', 'duoc_thu', 'dtqgvn', 'thuoc'
    ]
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_chunk_size: int = 50,
        max_chunk_size: int = 3000,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        self.splitter = MarkdownHeaderTextSplitter()
        self.garbage_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in PDF_GARBAGE_PATTERNS]
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'chars_processed': 0,
            'font_corrections_made': 0,
            'garbage_lines_removed': 0,
            'medical_terms_fixed': 0,
            'drugs_detected': 0,
        }
    
    def _is_drug_document(self, file_path: Path) -> bool:
        """
        Kiểm tra xem file có thuộc Dược thư không.
        """
        path_str = str(file_path).lower()
        return any(folder in path_str for folder in self.DRUG_DOCUMENT_FOLDERS)
    
    def fix_font_errors(self, text: str) -> str:
        """
        Sửa lỗi font hệ thống.
        
        Quy tắc:
        1. Thay thế các từ cụ thể trong FONT_CORRECTIONS
        2. Pattern tổng quát: ƣ -> ư (bao gồm cả các trường hợp không có trong dict)
        """
        corrected = text
        corrections_count = 0
        
        # Step 1: Thay thế các từ cụ thể (để đảm bảo chính xác ngữ cảnh)
        for wrong, correct in FONT_CORRECTIONS.items():
            if wrong in corrected:
                count = corrected.count(wrong)
                corrections_count += count
                corrected = corrected.replace(wrong, correct)
        
        # Step 2: Pattern tổng quát cho ƣ -> ư (bao phủ các trường hợp còn sót)
        # Thay thế tất cả ƣ còn lại thành ư
        if 'ƣ' in corrected:
            count = corrected.count('ƣ')
            corrections_count += count
            corrected = corrected.replace('ƣ', 'ư')
        if 'Ƣ' in corrected:
            count = corrected.count('Ƣ')
            corrections_count += count
            corrected = corrected.replace('Ƣ', 'Ư')
        
        self.stats['font_corrections_made'] += corrections_count
        return corrected
    
    def fix_medical_terms_in_text(self, text: str) -> str:
        """
        Áp dụng fix_medical_terms và đếm số lỗi đã sửa.
        """
        original = text
        fixed = fix_medical_terms(text)
        
        # Đếm số thay đổi (ước tính)
        if original != fixed:
            # Đếm số khác biệt
            diff_count = sum(1 for a, b in zip(original, fixed) if a != b)
            self.stats['medical_terms_fixed'] += max(1, diff_count // 10)
        
        return fixed
    
    def remove_pdf_garbage(self, text: str) -> str:
        """
        Loại bỏ rác PDF bao gồm:
        - DTQGVN và các biến thể
        - Số trang đơn lẻ
        - Ký tự lặp lại vô nghĩa
        - Header/footer decorations
        """
        lines = text.split('\n')
        clean_lines = []
        removed_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Fast checks first (no regex)
            # 1. Dòng chứa DTQGVN
            if 'DTQGVN' in stripped.upper():
                removed_count += 1
                continue
            
            # 2. Dòng chỉ có số (số trang)
            if stripped.isdigit() and len(stripped) <= 4:
                removed_count += 1
                continue
            
            # 3. Dòng rỗng hoặc quá ngắn không cần check regex
            if len(stripped) < 3:
                clean_lines.append(line)
                continue
            
            # 4. Dòng chỉ có ký tự đặc biệt lặp lại
            if len(stripped) > 5:
                non_space = stripped.replace(' ', '')
                if non_space and not any(c.isalnum() for c in non_space):
                    unique_chars = set(non_space)
                    if len(unique_chars) <= 3:
                        removed_count += 1
                        continue
            
            # 5. Regex checks (only if needed)
            is_garbage = False
            for pattern in self.garbage_patterns:
                if pattern.match(stripped):
                    is_garbage = True
                    removed_count += 1
                    break
            
            if not is_garbage:
                clean_lines.append(line)
        
        self.stats['garbage_lines_removed'] += removed_count
        return '\n'.join(clean_lines)
    
    def normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa văn bản.
        
        Quy tắc:
        1. Đảm bảo có dòng trống sau mỗi tiêu đề (#, ##, ###, ...)
        2. Giữ nguyên dấu gạch đầu dòng (-)
        3. Xóa nhiều dòng trống liên tiếp
        4. Loại bỏ khoảng trắng thừa
        """
        # Step 1: Đảm bảo có dòng trống SAU mỗi heading
        # Pattern: dòng heading (#...) theo sau bởi dòng không trống (không phải dòng trống)
        text = re.sub(r'^(#{1,6}\s+[^\n]+)\n(?!\n)(?!#)', r'\1\n\n', text, flags=re.MULTILINE)
        
        # Step 2: Thay thế nhiều dòng trống liên tiếp (>2) thành 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 3: Xóa khoảng trắng thừa ở cuối mỗi dòng
        text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
        
        # Step 4: Chuẩn hóa bullet points - giữ nguyên dấu gạch đầu dòng
        text = re.sub(r'^[-•]\s*', '- ', text, flags=re.MULTILINE)
        
        # Step 5: Xử lý bảng và LaTeX
        text = self.normalize_tables(text)
        text = self.normalize_latex(text)
        
        return text.strip()
    
    def normalize_tables(self, text: str) -> str:
        """Chuẩn hóa bảng Markdown."""
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            if '|' in line and not line.strip().startswith('#'):
                if re.match(r'^\s*\|?\s*[-:]+\s*\|', line):
                    normalized_lines.append(line)
                else:
                    cells = line.split('|')
                    normalized_cells = []
                    for i, cell in enumerate(cells):
                        if i == 0 and cell.strip() == '':
                            normalized_cells.append('')
                        elif i == len(cells) - 1 and cell.strip() == '':
                            normalized_cells.append('')
                        else:
                            content = cell.strip()
                            if content:
                                normalized_cells.append(f' {content} ')
                            else:
                                normalized_cells.append(cell)
                    normalized_lines.append('|'.join(normalized_cells))
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def normalize_latex(self, text: str) -> str:
        """Chuẩn hóa biểu thức LaTeX."""
        text = re.sub(r'([^\s\n\$])\$\$', r'\1 $$', text)
        text = re.sub(r'\$\$([^\s\n\$])', r'$$ \1', text)
        text = re.sub(r'([^\s\n\$])\$(?!\$)', r'\1 $', text)
        text = re.sub(r'(?<!\$)\$([^\s\n\$])', r'$ \1', text)
        return text
    
    def create_context_prefix(
        self, 
        file_name: str, 
        headers: List[str], 
        subject_name: str = None,
        atc_code: str = None,
        is_drug: bool = False,
        hierarchy_state: HierarchyState = None
    ) -> str:
        """
        Tạo prefix context injection theo format yêu cầu.
        
        Format mới: [BỆNH: {disease_name} | MỤC: {h1} > {h2} > {h3}]
        Hoặc với thuốc: [MÃ ATC: {code} | THUỐC: {Tên} | MỤC: {h1} > {h2} > {h3}]
        
        Args:
            file_name: Tên file gốc
            headers: Danh sách các tiêu đề theo thứ bậc
            subject_name: Tên thuốc/bệnh
            atc_code: Mã ATC (nếu có)
            is_drug: True nếu là thuốc, False nếu là bệnh
            hierarchy_state: Trạng thái phân cấp tiêu đề (nếu có)
        
        Returns:
            String prefix theo format chuẩn
        """
        # Nếu có hierarchy_state, sử dụng nó
        if hierarchy_state and hierarchy_state.disease_name:
            context = hierarchy_state.get_context()
            
            # Thêm mã ATC nếu có (cho thuốc) - chỉ thêm mã, không thêm nhãn
            if atc_code and is_drug:
                # Chèn mã ATC vào đầu
                context = context.replace('[', f'[{atc_code} | ')
            
            return context
        
        # Fallback: Logic cũ nếu không có hierarchy_state
        
        # Xác định tên subject
        if subject_name:
            name = self._fix_protocol_name(subject_name)
        elif headers:
            name = self._fix_protocol_name(headers[0])
        else:
            name = self._format_file_name_as_protocol(file_name)
        
        # Xây dựng hierarchy từ headers
        parts = []
        
        # Ưu tiên mã ATC lên đầu nếu có (giữ lại nhưng không có nhãn)
        if atc_code:
            parts.append(atc_code)
        
        # Thêm tên (không có nhãn BỆNH: hay THUỐC:)
        parts.append(name)
        
        # Thêm hierarchy đầy đủ (không có nhãn MỤC:)
        if headers and len(headers) > 0:
            fixed_headers = [self._fix_protocol_name(h) for h in headers]
            # Bỏ tên bệnh/thuốc (header đầu tiên) nếu trùng với name
            if fixed_headers and fixed_headers[0].upper() == name.upper():
                fixed_headers = fixed_headers[1:]
            
            if fixed_headers:
                muc_hierarchy = ' > '.join(fixed_headers)
                parts.append(muc_hierarchy)
        
        prefix = " | ".join(parts)
        return f"[{prefix}]"
    
    def _fix_protocol_name(self, name: str) -> str:
        """
        Sửa lỗi và làm sạch tên tiêu đề/protocol.
        
        Quy tắc:
        1. Loại bỏ ký tự trang trí: **, *, _, #
        2. Sửa lỗi mất chữ đầu
        3. Sửa lỗi font ƣ -> ư
        4. Loại bỏ khoảng trắng thừa
        """
        if not name:
            return name
        
        # Step 1: Loại bỏ ký tự trang trí **, *, _
        cleaned = name.strip()
        cleaned = re.sub(r'^[\*_#]+\s*', '', cleaned)  # Đầu dòng
        cleaned = re.sub(r'\s*[\*_#]+$', '', cleaned)  # Cuối dòng
        cleaned = re.sub(r'\*{2,}', '', cleaned)  # ** trong dòng
        cleaned = re.sub(r'_{2,}', '', cleaned)  # __ trong dòng
        
        # Step 2: Sửa lỗi font ƣ -> ư
        for wrong, correct in FONT_CORRECTIONS.items():
            if wrong in cleaned:
                cleaned = cleaned.replace(wrong, correct)
        # Fallback cho ƣ còn sót
        cleaned = cleaned.replace('ƣ', 'ư').replace('Ƣ', 'Ư')
        
        # Step 3: Sửa lỗi mất chữ đầu
        for wrong, correct in MEDICAL_TERM_CORRECTIONS.items():
            if cleaned.lower() == wrong.lower():
                return correct
            if cleaned.lower().startswith(wrong.lower() + ' '):
                return correct + cleaned[len(wrong):]
            if cleaned.lower().startswith(wrong.lower()):
                remaining = cleaned[len(wrong):]
                if not remaining or remaining[0] in ' :.,;!':
                    return correct + remaining
        
        # Step 4: Loại bỏ khoảng trắng thừa
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _format_file_name_as_protocol(self, file_name: str) -> str:
        """Chuyển đổi tên file thành tên protocol."""
        name = Path(file_name).stem
        name = name.replace('_', ' ').replace('-', ' ')
        name = name.title()
        return name
    
    def split_large_chunk(self, content: str, max_size: int) -> List[str]:
        """Chia chunk lớn thành các phần nhỏ hơn."""
        if len(content) <= max_size:
            return [content]
        
        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(para) > max_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= max_size:
                            current_chunk = current_chunk + " " + sent if current_chunk else sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_file(self, file_path: Path) -> List[MasterChunk]:
        """Xử lý một file markdown với HierarchyState để đảm bảo kế thừa tiêu đề."""
        print(f"  📄 Đang xử lý: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.stats['chars_processed'] += len(content)
        
        # Step 1: Fix font errors
        content = self.fix_font_errors(content)
        
        # Step 2: Fix medical terms (sửa lỗi mất chữ)
        content = self.fix_medical_terms_in_text(content)
        
        # Step 3: Remove PDF garbage (DTQGVN, số trang, noise)
        content = self.remove_pdf_garbage(content)
        
        # Step 4: Normalize text
        content = self.normalize_text(content)
        
        # Step 5: Xác định loại document
        is_drug_doc = self._is_drug_document(file_path)
        
        # Step 6: Xử lý line-by-line với HierarchyState
        file_name = file_path.stem
        master_chunks = self._process_with_hierarchy(
            content=content,
            file_name=file_name,
            file_path=file_path,
            is_drug_doc=is_drug_doc
        )
        
        print(f"    ✓ Tạo {len(master_chunks)} chunks" + 
              (f" ({self.stats['drugs_detected']} thuốc)" if is_drug_doc else ""))
        
        # Merge các chunk nhỏ hơn 150 ký tự
        original_count = len(master_chunks)
        master_chunks = self._merge_tiny_chunks(master_chunks)
        merged_count = original_count - len(master_chunks)
        if merged_count > 0:
            print(f"    ⚡ Đã gộp {merged_count} chunks nhỏ → {len(master_chunks)} chunks")
        
        return master_chunks
    
    def _merge_tiny_chunks(self, chunks: List[MasterChunk], min_chars: int = 150) -> List[MasterChunk]:
        """
        Gộp các chunk nhỏ hơn min_chars ký tự vào chunk tiếp theo.
        
        Quy tắc:
        - Nếu chunk < min_chars, gộp vào chunk sau
        - Khi gộp: thêm các tiêu đề phân cấp trước đó vào chunk sau
        - Format: [Tiêu đề bệnh] > [H1] > [H2] > [H3] + nội dung chunk nhỏ + nội dung chunk sau
        """
        if not chunks:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_content = current_chunk.content
            current_char_count = len(current_content)
            
            # Nếu chunk hiện tại đủ lớn, giữ nguyên
            if current_char_count >= min_chars:
                merged.append(current_chunk)
                i += 1
                continue
            
            # Chunk nhỏ - cần gộp vào chunk sau
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                # Lấy hierarchy từ chunk nhỏ để thêm vào đầu chunk sau
                small_hierarchy = current_chunk.metadata.get('heading_hierarchy', [])
                next_hierarchy = next_chunk.metadata.get('heading_hierarchy', [])
                
                # Tạo prefix từ hierarchy của chunk nhỏ
                hierarchy_prefix = ""
                if small_hierarchy:
                    # Chỉ lấy các heading KHÁC với next_chunk để tránh lặp
                    unique_headers = []
                    for h in small_hierarchy:
                        if h not in next_hierarchy:
                            unique_headers.append(h)
                    
                    if unique_headers:
                        hierarchy_prefix = " > ".join(unique_headers) + "\n\n"
                
                # Gộp nội dung: hierarchy_prefix + chunk nhỏ + chunk sau
                # Loại bỏ context prefix cũ từ cả 2 chunks để tránh duplicate
                small_content_only = self._extract_raw_content(current_content)
                next_content_only = self._extract_raw_content(next_chunk.content)
                
                # Tạo context prefix mới cho chunk gộp
                merged_context_prefix = self.create_context_prefix(
                    file_name=next_chunk.metadata.get('file_name', ''),
                    headers=next_hierarchy,
                    subject_name=next_chunk.metadata.get('subject_name', ''),
                    atc_code=next_chunk.metadata.get('atc_code'),
                    is_drug=next_chunk.metadata.get('subject_type') == 'drug',
                    hierarchy_state=None  # Sẽ build từ metadata
                )
                
                # Nội dung gộp
                merged_content = f"{merged_context_prefix}\n\n"
                if hierarchy_prefix:
                    merged_content += f"{hierarchy_prefix}"
                merged_content += f"{small_content_only}\n\n{next_content_only}"
                
                # Cập nhật metadata cho chunk gộp
                merged_metadata = next_chunk.metadata.copy()
                merged_metadata['char_count'] = len(merged_content)
                merged_metadata['word_count'] = len(merged_content.split())
                merged_metadata['merged_from_small_chunk'] = True
                merged_metadata['original_small_chunk_size'] = current_char_count
                
                # Tạo chunk mới đã gộp
                merged_chunk = MasterChunk(
                    id=next_chunk.id,
                    content=merged_content,
                    metadata=merged_metadata
                )
                
                merged.append(merged_chunk)
                i += 2  # Skip cả 2 chunks (đã gộp)
            else:
                # Đây là chunk cuối cùng và nó nhỏ - giữ lại hoặc gộp vào chunk trước
                if merged:
                    # Gộp vào chunk trước
                    prev_chunk = merged[-1]
                    prev_content_only = self._extract_raw_content(prev_chunk.content)
                    small_content_only = self._extract_raw_content(current_content)
                    
                    # Tạo context prefix mới
                    merged_context_prefix = self.create_context_prefix(
                        file_name=prev_chunk.metadata.get('file_name', ''),
                        headers=prev_chunk.metadata.get('heading_hierarchy', []),
                        subject_name=prev_chunk.metadata.get('subject_name', ''),
                        atc_code=prev_chunk.metadata.get('atc_code'),
                        is_drug=prev_chunk.metadata.get('subject_type') == 'drug',
                        hierarchy_state=None
                    )
                    
                    new_content = f"{merged_context_prefix}\n\n{prev_content_only}\n\n{small_content_only}"
                    
                    prev_chunk.content = new_content
                    prev_chunk.metadata['char_count'] = len(new_content)
                    prev_chunk.metadata['word_count'] = len(new_content.split())
                    prev_chunk.metadata['merged_with_last_small_chunk'] = True
                else:
                    # Không có chunk nào trước đó, giữ lại chunk nhỏ này
                    merged.append(current_chunk)
                
                i += 1
        
        return merged
    
    def _extract_raw_content(self, full_content: str) -> str:
        """
        Trích xuất nội dung thô từ content đã có context prefix.
        Loại bỏ phần [BỆNH: ... | MỤC: ...] ở đầu.
        """
        lines = full_content.split('\n')
        
        # Tìm dòng đầu tiên không phải context prefix (không bắt đầu bằng [)
        content_start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('['):
                content_start_idx = i
                break
        
        # Trả về nội dung từ dòng đó trở đi
        return '\n'.join(lines[content_start_idx:]).strip()
    
    def _process_with_hierarchy(
        self, 
        content: str, 
        file_name: str, 
        file_path: Path,
        is_drug_doc: bool
    ) -> List[MasterChunk]:
        """
        Xử lý văn bản với HierarchyState để đảm bảo kế thừa tiêu đề xuyên suốt.
        
        Logic:
        1. Tìm tiêu đề bệnh chính (# TÊN BỆNH) - LUÔN kiểm tra, không chỉ lần đầu
        2. Khi gặp bệnh mới → RESET toàn bộ hierarchy + đặt disease_name mới
        3. Khi gặp TÀI LIỆU THAM KHẢO → skip nội dung nhưng TIẾP TỤC xử lý bệnh mới
        4. Tạo chunk với context injection đầy đủ
        """
        lines = content.split('\n')
        master_chunks = []
        
        # Khởi tạo HierarchyState
        hierarchy = HierarchyState()
        
        # Các biến tracking
        current_content_lines = []
        chunk_index = 0
        skip_reference_section = False
        first_heading_processed = False  # Track nếu đã xử lý heading đầu tiên
        
        for line in lines:
            stripped = line.strip()
            
            # ======= KIỂM TRA HEADING ĐẦU TIÊN CỦA FILE (#### TÊN BỆNH) =======
            # Nếu chưa có disease_name và gặp heading đầu tiên → đặt nó làm disease_name
            if not first_heading_processed and stripped.startswith('#'):
                first_heading_processed = True
                # Trích xuất nội dung heading
                heading_content = re.sub(r'^#+\s*', '', stripped)
                heading_content = re.sub(r'^\*{1,2}\s*', '', heading_content)
                heading_content = re.sub(r'\s*\*{1,2}$', '', heading_content)
                
                # Kiểm tra KHÔNG phải pattern số (1., I., a., ...) và KHÔNG phải tên tác giả
                # VÀ KHÔNG phải CHƯƠNG/PHẦN
                is_numbered = re.match(r'^(\d+\.|[IVXLCDM]+\.|[a-zA-Z][\.\)]|CHƯƠNG\s*[IVXLCDM\d]+|PHẦN\s*\d+)', heading_content, re.IGNORECASE)
                is_author = re.match(r'^(ThS\.|PGS\.|GS\.|TS\.|BS\.|BSCKII?\.|CN\.|KS\.)', heading_content, re.IGNORECASE)
                
                if not is_numbered and not is_author:
                    # Đây là tên bệnh đầu tiên
                    disease_name = self._fix_protocol_name(heading_content)
                    hierarchy.set_disease_name(disease_name)
                    current_content_lines.append(line)
                    continue
            
            # ======= KIỂM TRA TIÊU ĐỀ BỆNH MỚI (# TÊN BỆNH) - LUÔN KIỂM TRA =======
            # Điều kiện: dòng bắt đầu bằng # (không phải ##, ###, ...)
            if stripped.startswith('# ') and not stripped.startswith('## '):
                content_after_hash = stripped[2:].strip()
                # Loại bỏ dấu ** nếu có
                content_after_hash = re.sub(r'^\*{1,2}\s*', '', content_after_hash)
                content_after_hash = re.sub(r'\s*\*{1,2}$', '', content_after_hash)
                
                # Kiểm tra KHÔNG phải pattern tiêu đề số (1., I., a., ...)
                # VÀ KHÔNG phải TÀI LIỆU THAM KHẢO
                # VÀ KHÔNG phải tên tác giả
                # VÀ KHÔNG phải pattern "Bước X:", "Bảng X:", "Hình X:", "PHẦN X:", "CHƯƠNG X"
                is_numbered = re.match(r'^(\d+\.|[IVXLCDM]+\.|[a-zA-Z]\.|BỆNH:|Bước\s*\d+|Bảng\s*\d+|Hình\s*\d+|PHẦN\s*\d+|CHƯƠNG\s*[IVXLCDM\d]+)', content_after_hash, re.IGNORECASE)
                is_reference = self._is_reference_heading(content_after_hash)
                is_author = re.match(r'^(ThS\.|PGS\.|GS\.|TS\.|BS\.|BSCKII?\.|CN\.|KS\.)', content_after_hash, re.IGNORECASE)
                
                # Kiểm tra thêm: KHÔNG phải heading phụ như "Cận lâm sàng:", "Điều trị:", ...
                # Các heading phụ thường kết thúc bằng dấu ":" hoặc chỉ 1-2 từ
                is_subsection = (
                    content_after_hash.endswith(':') or  # Kết thúc bằng ":"
                    len(content_after_hash.split()) <= 2 or  # Chỉ 1-2 từ
                    content_after_hash.upper() in ['ĐẠI CƯƠNG', 'NGUYÊN NHÂN', 'TRIỆU CHỨNG', 
                        'CHẨN ĐOÁN', 'ĐIỀU TRỊ', 'TIÊN LƯỢNG', 'PHÒNG NGỪA', 'DỰ PHÒNG',
                        'CẬN LÂM SÀNG', 'LÂM SÀNG', 'BIẾN CHỨNG', 'PHÂN LOẠI']
                )
                
                if not is_numbered and not is_reference and not is_subsection and not is_author and len(content_after_hash) > 8:
                    # ĐÂY LÀ BỆNH MỚI!
                    
                    # Lưu chunk hiện tại (của bệnh cũ) nếu có
                    if current_content_lines and not skip_reference_section:
                        chunk = self._create_chunk_from_lines(
                            lines=current_content_lines,
                            hierarchy=hierarchy,
                            file_name=file_name,
                            file_path=file_path,
                            chunk_index=chunk_index,
                            is_drug=is_drug_doc
                        )
                        if chunk:
                            master_chunks.append(chunk)
                            chunk_index += 1
                        current_content_lines = []
                    
                    # RESET TOÀN BỘ HIERARCHY cho bệnh mới
                    hierarchy.reset()
                    disease_name = self._fix_protocol_name(content_after_hash)
                    hierarchy.set_disease_name(disease_name)
                    
                    # Tắt skip reference (bệnh mới bắt đầu)
                    skip_reference_section = False
                    
                    current_content_lines.append(line)
                    continue
            
            # ======= KIỂM TRA TÀI LIỆU THAM KHẢO =======
            # Kiểm tra các heading ### TÀI LIỆU THAM KHẢO
            if stripped.startswith('#'):
                # Trích xuất nội dung heading
                heading_content = re.sub(r'^#+\s*', '', stripped)
                heading_content = re.sub(r'^\*{1,2}\s*', '', heading_content)
                heading_content = re.sub(r'\s*\*{1,2}$', '', heading_content)
                
                if self._is_reference_heading(heading_content):
                    # Lưu chunk hiện tại trước khi skip
                    if current_content_lines and not skip_reference_section:
                        chunk = self._create_chunk_from_lines(
                            lines=current_content_lines,
                            hierarchy=hierarchy,
                            file_name=file_name,
                            file_path=file_path,
                            chunk_index=chunk_index,
                            is_drug=is_drug_doc
                        )
                        if chunk:
                            master_chunks.append(chunk)
                            chunk_index += 1
                        current_content_lines = []
                    
                    # Bắt đầu skip phần tài liệu tham khảo
                    skip_reference_section = True
                    continue
            
            # Nếu đang skip reference section
            if skip_reference_section:
                # Kiểm tra xem có phải bệnh mới không (đã xử lý ở trên)
                # Nếu không phải heading # TÊN BỆNH thì tiếp tục skip
                continue
            
            # ======= NHẬN DIỆN CẤP ĐỘ MARKDOWN HEADING (#, ##, ###, ...) =======
            if stripped.startswith('#'):
                # Đếm số lượng # để xác định cấp độ markdown
                match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
                if match:
                    hash_count = len(match.group(1))  # Số lượng #
                    heading_content = match.group(2).strip()
                    
                    # Loại bỏ dấu ** nếu có
                    heading_content = re.sub(r'^\*{1,2}\s*', '', heading_content)
                    heading_content = re.sub(r'\s*\*{1,2}$', '', heading_content)
                    
                    # Fix tên tiêu đề
                    heading_content = self._fix_protocol_name(heading_content)
                    
                    # Lưu chunk hiện tại trước khi cập nhật hierarchy
                    if current_content_lines:
                        chunk = self._create_chunk_from_lines(
                            lines=current_content_lines,
                            hierarchy=hierarchy,
                            file_name=file_name,
                            file_path=file_path,
                            chunk_index=chunk_index,
                            is_drug=is_drug_doc
                        )
                        if chunk:
                            master_chunks.append(chunk)
                            chunk_index += 1
                        current_content_lines = []
                    
                    # Cập nhật HierarchyState theo cấp độ markdown (#=1, ##=2, ###=3, ...)
                    hierarchy.update_heading(hash_count, heading_content)
                    
                    current_content_lines.append(line)
                    continue
            
            # Không phải heading - thêm vào content
            current_content_lines.append(line)
        
        # Chunk cuối cùng
        if current_content_lines and not skip_reference_section:
            chunk = self._create_chunk_from_lines(
                lines=current_content_lines,
                hierarchy=hierarchy,
                file_name=file_name,
                file_path=file_path,
                chunk_index=chunk_index,
                is_drug=is_drug_doc
            )
            if chunk:
                master_chunks.append(chunk)
        
        return master_chunks
    
    def _is_reference_heading(self, title: str) -> bool:
        """Kiểm tra xem tiêu đề có phải phần tài liệu tham khảo không."""
        # Chuẩn hóa text: loại bỏ khoảng trắng thừa giữa các ký tự
        normalized = title.upper().strip()
        # Loại bỏ khoảng trắng liên tiếp để match "T I LI U THAM" -> "TILIUTHAM"
        condensed = re.sub(r'\s+', '', normalized)
        
        reference_keywords = [
            'TÀI LIỆU THAM KHẢO', 'THAM KHẢO', 'REFERENCES', 'BIBLIOGRAPHY',
            'TAILIEUTHAMKHAO', 'TILIUTHAMKHAO', 'TILIUTHAMKHO', 'TILIUTHAMHAO',
            'TILIUTHAMHẢO',  # Variant with diacritic Ả
        ]
        
        # Check cả normalized và condensed version
        for kw in reference_keywords:
            kw_condensed = re.sub(r'\s+', '', kw)
            if kw in normalized or kw_condensed in condensed:
                return True
        
        # Fallback: check if condensed contains "THAMKH" hoặc "THAMH"
        if 'THAMKHẢO' in condensed or 'THAMKH' in condensed or 'THAMHẢO' in condensed:
            return True
        
        # Check pattern citation: "số. Tác giả (năm): ..." hoặc "số. Tác giả, *Tiêu đề*,..."
        # Pattern: bắt đầu bằng số + dấu chấm + chứa năm trong ngoặc hoặc chứa Nhà xuất bản
        if re.match(r'^\d+\.\s+', title):
            if re.search(r'\(\d{4}\)', title) or 'Nhà xuất bản' in title or 'xuất bản' in title.lower():
                return True
            # Pattern: có dấu * (italic) - thường là citation style
            if '*' in title and len(title) > 50:
                return True
        
        return False
    
    def _strip_reference_content(self, content: str) -> str:
        """
        Loại bỏ phần tài liệu tham khảo từ cuối nội dung chunk.
        
        Pattern phổ biến:
        - Danh sách đánh số: "1. Tác giả (năm): Tiêu đề..."
        - Danh sách dấu gạch: "- 1. Tác giả..."
        
        Returns:
            Content đã được loại bỏ phần tài liệu tham khảo
        """
        lines = content.split('\n')
        
        # Tìm điểm bắt đầu của phần tài liệu tham khảo
        ref_start_idx = None
        consecutive_refs = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Pattern: "- 1." hoặc "1." hoặc "2." đầu dòng + có dấu ngoặc (năm) hoặc chữ in nghiêng *...*
            is_ref_line = False
            
            # Pattern 1: "- 1. Tác giả (2010): ..." hoặc "1. Tác giả (2010): ..."
            if re.match(r'^[-•]?\s*\d+\.\s+\w+.*\(\d{4}\).*:', stripped):
                is_ref_line = True
            # Pattern 2: "- 1. Tác giả: *Tiêu đề*..." 
            elif re.match(r'^[-•]?\s*\d+\.\s+\w+.*\*.*\*', stripped):
                is_ref_line = True
            # Pattern 3: Dòng có "et al." hoặc "et al," (citation style)
            elif 'et al' in stripped.lower():
                is_ref_line = True
            # Pattern 4: Dòng có "Nhà xuất bản" hoặc "trang" + số
            elif 'Nhà xuất bản' in stripped or re.search(r'trang\s+\d+', stripped, re.IGNORECASE):
                is_ref_line = True
            
            if is_ref_line:
                consecutive_refs += 1
                if consecutive_refs >= 2 and ref_start_idx is None:
                    # Bắt đầu từ dòng ref đầu tiên (lùi lại)
                    ref_start_idx = i - (consecutive_refs - 1)
            else:
                # Reset nếu gặp dòng không phải tham khảo
                if not stripped:  # Dòng trống không reset
                    continue
                consecutive_refs = 0
        
        # Nếu tìm thấy phần tham khảo, cắt bỏ
        if ref_start_idx is not None:
            clean_lines = lines[:ref_start_idx]
            # Loại bỏ dòng trống cuối cùng
            while clean_lines and not clean_lines[-1].strip():
                clean_lines.pop()
            return '\n'.join(clean_lines)
        
        return content
    
    def _create_chunk_from_lines(
        self,
        lines: List[str],
        hierarchy: HierarchyState,
        file_name: str,
        file_path: Path,
        chunk_index: int,
        is_drug: bool
    ) -> Optional[MasterChunk]:
        """Tạo MasterChunk từ danh sách dòng với context injection."""
        
        # Skip chunk nếu đang trong phần tài liệu tham khảo
        if self._is_reference_heading(hierarchy.disease_name or ''):
            return None
        if any(self._is_reference_heading(h) for h in hierarchy.get_headers_list()):
            return None
        
        content = '\n'.join(lines).strip()
        
        # Loại bỏ phần tài liệu tham khảo nếu có
        content = self._strip_reference_content(content)
        
        # Skip chunk quá ngắn
        word_count = len(content.split())
        if word_count < 10:
            return None
        
        # Trích xuất mã ATC
        atc_code = extract_atc_code(content)
        
        # Tạo context prefix với HierarchyState
        context_prefix = self.create_context_prefix(
            file_name=file_name,
            headers=hierarchy.get_headers_list(),
            subject_name=hierarchy.disease_name,
            atc_code=atc_code,
            is_drug=is_drug,
            hierarchy_state=hierarchy
        )
        
        # Kết hợp prefix với content
        final_content = f"{context_prefix}\n\n{content}"
        
        # Chia nhỏ nếu quá dài
        if len(final_content) > self.max_chunk_size:
            # Chỉ trả về chunk đầu tiên, các chunk còn lại sẽ được xử lý riêng
            sub_contents = self.split_large_chunk(content, self.max_chunk_size - len(context_prefix) - 10)
            final_content = f"{context_prefix}\n\n{sub_contents[0]}"
        
        # Tạo chunk ID
        chunk_id = f"{file_name}_{chunk_index:04d}"
        
        # Tạo metadata
        metadata = {
            'source_file': str(file_path),
            'file_name': file_name,
            'subject_name': hierarchy.disease_name or self._format_file_name_as_protocol(file_name),
            'subject_type': 'drug' if is_drug else 'disease',
            'heading_hierarchy': hierarchy.get_headers_list(),
            'current_h1': hierarchy.h1,
            'current_h2': hierarchy.h2,
            'current_h3': hierarchy.h3,
            'atc_code': atc_code,
            'char_count': len(final_content),
            'word_count': len(final_content.split()),
            'chunk_index': chunk_index,
        }
        
        return MasterChunk(
            id=chunk_id,
            content=final_content,
            metadata=metadata
        )
    
    def _extract_protocol_name(self, chunks: List[Dict], file_name: str) -> str:
        """Trích xuất tên phác đồ."""
        if chunks and chunks[0]['metadata']['headers']:
            first_header = chunks[0]['metadata']['headers'][0]
            upper_header = first_header.upper()
            is_generic_keyword = any(
                upper_header == kw or upper_header.startswith(kw) 
                for kw in TOP_LEVEL_KEYWORDS
            )
            if not is_generic_keyword and len(first_header) > 5:
                return first_header
        
        return file_name.replace('_', ' ').title()
    
    def _merge_short_chunks(self, chunks: List[Dict], min_words: int = 50) -> List[Dict]:
        """Gộp các chunk ngắn."""
        if not chunks:
            return chunks
        
        merged = []
        pending_content = None
        pending_metadata = None
        pending_subject = None
        
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            metadata = chunk['metadata']
            subject_name = chunk.get('subject_name')
            word_count = len(content.split())
            
            if pending_content:
                current_headers = metadata.get('headers', [])
                pending_headers = pending_metadata.get('headers', [])
                
                # KHÔNG gộp nếu khác subject (thuốc/bệnh khác nhau)
                same_subject = (pending_subject == subject_name) or (not pending_subject) or (not subject_name)
                
                same_parent = (
                    len(current_headers) > 0 and 
                    len(pending_headers) > 0 and
                    current_headers[:-1] == pending_headers[:-1]
                ) if len(current_headers) > 1 and len(pending_headers) > 1 else True
                
                pending_word_count = len(pending_content.split())
                should_merge = same_subject and (same_parent or pending_word_count < 20)
                
                if should_merge:
                    content = pending_content + "\n\n" + content
                    if len(pending_headers) <= len(current_headers):
                        metadata = {
                            'headers': pending_headers,
                            'header_levels': pending_metadata.get('header_levels', [])
                        }
                    subject_name = pending_subject or subject_name
                else:
                    merged.append({
                        'content': pending_content,
                        'metadata': pending_metadata,
                        'subject_name': pending_subject
                    })
                
                pending_content = None
                pending_metadata = None
                pending_subject = None
            
            current_word_count = len(content.split())
            
            if current_word_count < min_words and i < len(chunks) - 1:
                pending_content = content
                pending_metadata = metadata
                pending_subject = subject_name
            else:
                merged.append({
                    'content': content,
                    'metadata': metadata,
                    'subject_name': subject_name
                })
        
        if pending_content:
            if merged:
                # Chỉ gộp nếu cùng subject
                last_chunk = merged[-1]
                if last_chunk.get('subject_name') == pending_subject or not pending_subject:
                    last_chunk['content'] = last_chunk['content'] + "\n\n" + pending_content
                else:
                    merged.append({
                        'content': pending_content,
                        'metadata': pending_metadata,
                        'subject_name': pending_subject
                    })
            else:
                merged.append({
                    'content': pending_content,
                    'metadata': pending_metadata,
                    'subject_name': pending_subject
                })
        
        return merged
    
    def process_directory(self, recursive: bool = True) -> List[MasterChunk]:
        """Xử lý tất cả file markdown trong thư mục."""
        all_chunks = []
        
        pattern = '**/*.md' if recursive else '*.md'
        md_files = list(self.input_dir.glob(pattern))
        
        print(f"\n🔍 Tìm thấy {len(md_files)} file markdown\n")
        
        total_files = len(md_files)
        for idx, file_path in enumerate(sorted(md_files), 1):
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
                self.stats['files_processed'] += 1
                
                # Progress indicator every 50 files
                if idx % 50 == 0:
                    print(f"  📊 Tiến độ: {idx}/{total_files} files ({idx*100//total_files}%)")
                    
            except Exception as e:
                print(f"    ✗ Lỗi: {e}")
        
        self.stats['chunks_created'] = len(all_chunks)
        return all_chunks
    
    def save_to_jsonl(self, chunks: List[MasterChunk], output_file: str = None) -> str:
        """Lưu chunks vào file JSONL."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"medical_master_data_{timestamp}.jsonl"
        
        output_path = self.output_dir / output_file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.to_json() + '\n')
        
        return str(output_path)
    
    def print_statistics(self):
        """In thống kê xử lý."""
        print("\n" + "=" * 60)
        print("📊 THỐNG KÊ XỬ LÝ")
        print("=" * 60)
        print(f"📁 Số file đã xử lý:        {self.stats['files_processed']:,}")
        print(f"📄 Số chunks đã tạo:        {self.stats['chunks_created']:,}")
        print(f"📝 Tổng ký tự đã xử lý:     {self.stats['chars_processed']:,}")
        print(f"🔧 Số lỗi font đã sửa:      {self.stats['font_corrections_made']:,}")
        print(f"💊 Số thuật ngữ y khoa sửa: {self.stats['medical_terms_fixed']:,}")
        print(f"🗑️  Số dòng rác đã loại:     {self.stats['garbage_lines_removed']:,}")
        print(f"💉 Số thuốc phát hiện:      {self.stats['drugs_detected']:,}")
        print("=" * 60)


def display_sample_chunks(chunks: List[MasterChunk], sample_size: int = 20):
    """Hiển thị mẫu chunks ngẫu nhiên."""
    print("\n" + "=" * 80)
    print(f"🎲 HIỂN THỊ {min(sample_size, len(chunks))} CHUNKS NGẪU NHIÊN")
    print("=" * 80)
    
    sample = random.sample(chunks, min(sample_size, len(chunks)))
    
    for i, chunk in enumerate(sample, 1):
        print(f"\n{'─' * 80}")
        print(f"📄 CHUNK #{i}")
        print(f"{'─' * 80}")
        print(f"🆔 ID: {chunk.id}")
        print(f"📂 File: {chunk.metadata.get('file_name', 'N/A')}")
        print(f"🏷️  Subject: {chunk.metadata.get('subject_name', 'N/A')} ({chunk.metadata.get('subject_type', 'N/A')})")
        print(f"🔗 Hierarchy: {' > '.join(chunk.metadata.get('heading_hierarchy', []))}")
        if chunk.metadata.get('atc_code'):
            print(f"💊 ATC Code: {chunk.metadata.get('atc_code')}")
        print(f"📊 Chars: {chunk.metadata.get('char_count', 0):,} | Words: {chunk.metadata.get('word_count', 0):,}")
        print(f"{'─' * 40}")
        print("📝 NỘI DUNG:")
        
        preview = chunk.content[:600]
        if len(chunk.content) > 600:
            preview += "\n... [truncated]"
        print(preview)
    
    print("\n" + "=" * 80)


def main():
    """Hàm main để chạy converter."""
    
    # Cấu hình đường dẫn
    INPUT_DIR = "/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot/rehierarchy_output"
    OUTPUT_DIR = "/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot/data/chunks"
    OUTPUT_FILE = "medical_master_data6.jsonl"
    
    print("=" * 80)
    print("🏥 MEDICAL MARKDOWN TO JSONL CONVERTER v3.0")
    print("   Tích hợp Hierarchy Processor - Kế thừa phân cấp tiêu đề xuyên suốt")
    print("=" * 80)
    print(f"📁 Input:  {INPUT_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"📄 File:   {OUTPUT_FILE}")
    print("=" * 80)
    print("\n🔧 CẢI TIẾN v3.0:")
    print("   ✓ HierarchyState: Quản lý disease_name, h1, h2, h3 xuyên suốt")
    print("   ✓ Context Injection: [BỆNH: {name} | MỤC: {h1} > {h2} > {h3}]")
    print("   ✓ Detect Heading Level: Regex chuẩn nhận diện cấp tiêu đề")
    print("   ✓ Giới hạn độ dài tiêu đề (≤200 ký tự) tránh nhận nhầm")
    print("   ✓ Fix Medical Terms (sửa lỗi mất chữ đầu)")
    print("   ✓ Noise Removal (DTQGVN, số trang, ký tự lặp)")
    print("   ✓ Reference Section Skip (bỏ qua TÀI LIỆU THAM KHẢO)")
    print("=" * 80)
    
    # Khởi tạo converter
    converter = MedicalMarkdownConverter(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        min_chunk_size=50,
        max_chunk_size=3000,
    )
    
    # Xử lý tất cả file
    print("\n🚀 BẮT ĐẦU XỬ LÝ...")
    chunks = converter.process_directory(recursive=True)
    
    if not chunks:
        print("\n⚠️ Không tạo được chunk nào!")
        return
    
    # Lưu vào JSONL
    output_path = converter.save_to_jsonl(chunks, OUTPUT_FILE)
    print(f"\n✅ Đã lưu vào: {output_path}")
    
    # In thống kê
    converter.print_statistics()
    
    # Hiển thị mẫu chunks
    display_sample_chunks(chunks, sample_size=20)
    
    # Thông tin file output
    file_size = os.path.getsize(output_path)
    print(f"\n📦 Kích thước file output: {file_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
