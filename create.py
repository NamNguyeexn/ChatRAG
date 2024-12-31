import logging
from flask import Flask, request, jsonify
from google import genai

# Khởi tạo ứng dụng Flask và Google Gemini Client
app = Flask(__name__)
gemini_api_key = "AIzaSyATMQ3R0VgO4JypfpdhYa6ejsul_zuaybU"  # Thay bằng API Key thực của bạn
client = genai.Client(api_key=gemini_api_key)
model = 'gemini-2.0-flash-exp'
@app.route('/label', methods=['POST'])
def generate_labels():
    try:
        # Nhận dữ liệu từ yêu cầu Spring
        data = request.get_json()
        if not data or 'title' not in data:
            logging.warning("Yêu cầu không hợp lệ: Thiếu trường 'title'")
            return jsonify({"error": "Trường 'title' không được để trống"}), 400

        title = data['title'].strip()
        if not title:
            logging.warning("Trường 'title' trống")
            return jsonify({"error": "Trường 'title' không được để trống"}), 400

        logging.info(f"Nhận product_title: {title}")

        # Tạo prompt để gọi API Gemini
        prompt = (
            f"Từ đoạn văn bản sau: '{title}', hãy trích xuất và trả về các nhãn liên quan. "
            f"Ngắn gọn với ví dụ như 'quần', 'áo', 'giày', 'dép', 'nam', 'nữ'. "
            f"Yêu cầu 2 nhãn liên quan đến tình huống sử dụng, 3 nhãn liên quan đến tính từ mô tả, "
            f"3 nhãn liên quan đến đặc điểm, 1 nhãn liên quan đến giới tính \
            và các nhãn liên quan đến màu sắc. Tất cả các nhãn không được chứa mùa. "
            f"Chỉ trả về danh sách các nhãn, không bao gồm thông tin nào khác, \
            Không chứa kí tự '[' ']' và các dấu ', chỉ chứa dấu phẩy ',' ."
        )

        # Gọi API Gemini để sinh nội dung
        response = client.models.generate_content(
            model= model, contents=prompt
        )

        # Phân tích phản hồi từ Gemini
        generated_text = response.text.strip()
        print(f"Kết quả từ Gemini: {generated_text}")

        # Tách nhãn từ kết quả
        generated_labels = generated_text.split(",")
        if not generated_labels:
            logging.warning("Không nhận được nhãn nào từ Gemini")
            return jsonify({"error": "Không tạo được nhãn phù hợp"}), 400

        # Gửi lại nhãn cho Spring
        print(f"Kết quả gửi đi: {generated_labels}")
        return jsonify({"labels": generated_labels}), 200

    except Exception as e:
        logging.error(f"Lỗi khi xử lý yêu cầu: {str(e)}", exc_info=True)
        return jsonify({"error": "Đã xảy ra lỗi trong quá trình xử lý"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
