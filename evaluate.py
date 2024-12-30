import csv

# Đọc file TXT
input_file = "output-onlinetexttools.txt"

with open(input_file, "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip()]

# Xử lý văn bản để tạo danh sách câu hỏi và đáp án
questions = []

for i in range(0, len(lines), 6):  # Mỗi câu hỏi và các lựa chọn + đáp án chiếm 6 dòng
    try:
        question = lines[i:i+5]
        answer = lines[i + 5].split(": ")[1]  # Tách lấy phần đáp án sau "Đáp án: "
        questions.append((question, answer))
    except Exception as e:
        print(f"Error processing lines {i} to {i+5}: {e}")
        continue

# Tạo file CSV
output_file = "questions_answers2.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Câu hỏi", "Đáp án"])
    writer.writerows(questions)

print(f"File CSV đã được tạo: {output_file}")