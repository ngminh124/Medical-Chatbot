import os

# Cấu hình đường dẫn
base_dir = "/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot/data/output/duoc_thu_qg"
output_path = os.path.join(base_dir, "duoc_thu_qg_full.md")

with open(output_path, "w", encoding="utf-8") as outfile:
    for i in range(1, 8):
        file_path = f"{base_dir}/duoc_thu_qg_{i}/duoc_thu_qg_{i}.md"
        
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as infile:
                # Ghi tiêu đề phân đoạn
                outfile.write(f"\n\n# PHẦN {i}\n\n")
                outfile.write(infile.read())
                print(f"Đã nối xong tập {i}")
        else:
            print(f"Bỏ qua tập {i} vì không tìm thấy file.")

print(f"--- THÀNH CÔNG: File lưu tại {output_path} ---")