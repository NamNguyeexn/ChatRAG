from collections import Counter
from datetime import datetime
import faiss
import numpy as np
import pandas as pd
from faiss import IndexIDMap
from flask_caching import Cache
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sqlalchemy import URL, create_engine, text
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from underthesea import word_tokenize, pos_tag
# import py_vncorenlp
import logging

# Cho lan dau tien chay thi uncomment ham nay
# py_vncorenlp.download_model(save_dir='/home/namnguyeexn/Tai_lieu_hoc_tap/modelAI')

# model = py_vncorenlp.VnCoreNLP(save_dir='/home/namnguyeexn/Tai_lieu_hoc_tap/modelAI')

app = Flask(__name__)
# cache
app.config['CACHE_TYPE'] = 'RedisCache'
app.config['CACHE_REDIS_URL'] = 'redis://localhost:6379/0'
cache = Cache(app)

# GPT-4o Configuration
token = "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"
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

# Load Hugging Face embedding model
print("Đang tải mô hình embedding...")
embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Load FAISS index
print("Đang tải FAISS index...")
index = faiss.read_index(
    "/home/namnguyeexn/PycharmProjects/pythonProject/clothing_index.faiss")  # Tệp FAISS đã được tạo trước

if index.ntotal == 0:
    print(f"File FAISS index rỗng (không có vector).")
else:
    print(f"File FAISS index chứa {index.ntotal} vector.")

# Load danh sách câu trả lời (tương ứng với các vector trong FAISS)
answers = ["Trả lời 1", "Trả lời 2", "Trả lời 3"]  # Thay bằng danh sách của bạn

# Cấu hình Flask và database
DB_CONFIG = {
    "user": "root",
    "password": "123456aA@",
    "host": "localhost",
    "port": 3306,
    "database": "ecommerce"
}

FAISS_FILE = "/home/namnguyeexn/PycharmProjects/pythonProject/clothing_index.faiss"


def create_engine_connection():
    """
    Tạo kết nối SQLAlchemy tới MySQL.
    """
    db_config = URL.create(
        "mysql+pymysql",
        DB_CONFIG["user"],
        DB_CONFIG["password"],
        DB_CONFIG["host"],
        DB_CONFIG["port"],
        DB_CONFIG["database"]
    )
    return create_engine(db_config)


def load_data_from_db(engine, query):
    """
    Truy vấn dữ liệu từ database.
    """
    try:
        data = pd.read_sql(query, con=engine)
        if data.empty:
            raise ValueError("Không có dữ liệu nào được truy vấn từ bảng.")
        return data
    except Exception as e:
        raise ValueError(f"Lỗi khi tải dữ liệu: {e}")


def vectorize_data(data):
    """
    Vector hóa dữ liệu từ mô tả sản phẩm.
    """
    try:
        data['vector'] = data.apply(lambda row: embedding_model.encode(f"{row['title']} {row['description']}")
                                    .tolist(),
                                    axis=1)
        return data
    except Exception as e:
        raise ValueError(f"Lỗi khi vector hóa dữ liệu: {e}")


def create_faiss_index(data, faiss_file):
    try:
        vectors = np.array(data['vector'].tolist(), dtype='float32')
        ids = np.array(data['id'].tolist(), dtype='int64')

        index = IndexIDMap(faiss.IndexFlatL2(vectors.shape[1]))
        index.add_with_ids(vectors, ids)

        faiss.write_index(index, faiss_file)
        return True
    except Exception as e:
        raise ValueError(f"Lỗi khi tạo FAISS index: {e}")


@app.route("/initialize", methods=["GET"])
def initialize_index():
    """
    Endpoint để tải dữ liệu, vector hóa, và tạo FAISS index.
    """
    try:
        # Bước 1: Kết nối database
        engine = create_engine_connection()

        # Bước 2: Tải dữ liệu
        query = "SELECT id, score, create_at, update_at, price, signature, title, description FROM products"
        data = load_data_from_db(engine, query)

        # Bước 3: Vector hóa dữ liệu
        data = vectorize_data(data)

        # Bước 4: Tạo FAISS index
        create_faiss_index(data, FAISS_FILE)

        return jsonify({"message": "FAISS index đã được tạo thành công!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/add_product", methods=["POST"])
def add_product():
    """
    Thêm sản phẩm mới vào faiss
    """
    try:
        # Lấy dữ liệu từ request
        data = request.json

        # Kiểm tra dữ liệu
        required_fields = ["score", "create_at", "price", "update_at", "signature", "title", "description", "id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Trường '{field}' là bắt buộc"}), 400

        # Chuyển đổi dữ liệu
        product = {
            "id": data["id"],
            "score": float(data["score"]),
            "create_at": datetime.strptime(data["create_at"], "%Y-%m-%d %H:%M:%S"),
            "price": int(data["price"]),
            "update_at": datetime.strptime(data["update_at"], "%Y-%m-%d %H:%M:%S"),
            "signature": data["signature"],
            "title": data["title"],
            "description": data["description"]
        }

        # Bước 1: Lưu sản phẩm vào database
        engine = create_engine_connection()
        query = """
               INSERT INTO products (id, score, create_at, price, update_at, signature, title, description)
               VALUES (:id, :score, :create_at, :price, :update_at, :signature, :title, :description)
               """
        with engine.connect() as connection:
            connection.execute(query, product)

        # Bước 2: Vector hóa dữ liệu sản phẩm mới
        vector = embedding_model.encode(f"{product['title']} {product['description']}").astype('float32')

        # Bước 3: Thêm vector vào FAISS index
        index.add_with_ids(np.array([vector]), np.array([product["id"]], dtype='int64'))
        faiss.write_index(index, FAISS_FILE)

        return jsonify({"message": "Sản phẩm đã được thêm thành công và FAISS index đã được cập nhật."}), 200

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500


def extract_keywords_vietnamese(sentence):
    """
    Hàm để trích xuất danh từ và tính từ từ câu tiếng Việt.
    """
    try:
        tagged_words = pos_tag(sentence)  # Phân loại từ (POS tagging)
        # Lọc ra danh từ (N), danh từ riêng (Np) và tính từ (A)
        keywords = [word for word, tag in tagged_words if tag in ['N', 'Np', 'A']]
        return keywords
    except Exception as e:
        raise ValueError(f"Lỗi khi phân tích từ vựng: {e}")


@app.route('/query', methods=['POST'])
def handle_query():
    try:
        # Nhận câu hỏi từ request
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({"error": "Câu hỏi không được để trống"}), 400

        # Sử dụng GPT-4O để trích xuất từ khóa và xử lý câu hỏi
        print(f"Nhận câu hỏi: {question}")

        extracted_keywords = extract_keywords_vietnamese(question)
        print(f"Từ khóa được trích xuất: {extracted_keywords}")
        # Trả về danh sách các `name`
        existing_labels = fetch_existing_labels_from_db()
        # So khớp từ khóa trích xuất với nhãn đã có trong DB
        matching_keywords = [kw for kw in extracted_keywords if kw in existing_labels]

        # Kiểm tra nếu từ khóa đã đủ chi tiết
        if len(matching_keywords)/len(extracted_keywords) > 3/591:
            print("Từ khóa đủ phong phú, bỏ qua GPT.")
            relevant_adjectives = []  # Không gọi GPT
        else:
            # Cache kiểm tra để tránh gọi lại GPT
            cache_key = f"adjectives_{'_'.join(extracted_keywords)}"
            relevant_adjectives = cache.get(cache_key)

            if relevant_adjectives is None:
                # Gọi GPT để sinh tính từ liên quan
                gpt_response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system",
                         "content": "Bạn là một trợ lý AI giúp tạo danh sách các tính từ liên quan tới ngữ cảnh."},
                        {"role": "user",
                         "content": f"Sinh ra các tính từ mô tả sản phẩm phù hợp với các từ khóa: {', '.join(extracted_keywords)}"}
                    ]
                )
                generated_adjectives = gpt_response.choices[0].message.content.strip().split('\n')
                relevant_adjectives = []
                for line in generated_adjectives:
                    if not line.strip():
                        continue
                    if line.strip()[0].isdigit():
                        adjectives = line.split('. ')[1].strip().replace('**', '')
                        relevant_adjectives.append(adjectives)

                # Lưu vào cache
                cache.set(cache_key, relevant_adjectives, timeout=3600)  # Cache trong 1 giờ
            else:
                print("Dùng kết quả từ cache.")

        print(f"Tính từ được GPT sinh ra: {relevant_adjectives}")

        combined_keywords = extracted_keywords + relevant_adjectives
        print(f"Từ khóa kết hợp: {combined_keywords}")

        # 4. Tạo embedding từ các từ khóa kết hợp
        query_text = ' '.join(combined_keywords)
        question_embedding = embedding_model.encode(query_text).astype('float32').reshape(1, -1)

        # Tìm kiếm trong FAISS
        print("Tìm kiếm trong FAISS...")
        distances, indices = index.search(question_embedding, k=5)  # Tìm câu trả lời gần nhất (k=5)

        # Lấy thông tin sản phẩm từ database và xử lý LLM để tạo câu trả lời
        engine = create_engine_connection()
        query = "SELECT id, title, description, price FROM products WHERE id IN ({})".format(
            ', '.join(map(str, indices[0]))
        )
        products = load_data_from_db(engine, query)

        results = [int(product['id']) for _, product in products.iterrows()]
        return jsonify({
            "question": question,
            "results": results
        })
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Hàm kết nối cơ sở dữ liệu và lấy nhãn đã vector hóa
def fetch_existing_labels_from_db():
    try:
        # Kết nối cơ sở dữ liệu (thay đổi thông tin kết nối phù hợp)
        engine = create_engine_connection()
        with engine.connect() as conn:
            # Truy vấn lấy các label từ bảng
            query = text("SELECT name FROM labels")  # Lấy cột `name` hoặc `code` tùy theo yêu cầu
            result = conn.execute(query)
            existing_labels = [row[0] for row in result]
        return existing_labels
    except Exception as e:
        print(f"Lỗi khi lấy nhãn từ DB: {e}")
        return []


# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# @app.route('/label', methods=['POST'])
# def generate_labels():
#     try:
#         # Nhận dữ liệu từ Spring
#         data = request.get_json()
#         if not data or 'title' not in data:
#             logging.warning("Yêu cầu không hợp lệ: Thiếu trường 'title'")
#             return jsonify({"error": "Trường 'title' không được để trống"}), 400
#
#         title = data['title'].strip()
#         if not title:
#             logging.warning("Trường 'title' trống")
#             return jsonify({"error": "product_title không được để trống"}), 400
#
#         logging.info(f"Nhận product_title: {title}")
#
#         # Phân tích cú pháp với VnCoreNLP
#         annotated_sentences = model.annotate_text(title)
#         logging.debug(f"Thông tin phân tích cú pháp: {annotated_sentences}")
#
#         # Tạo nhãn dựa trên phân tích
#         generated_labels = set()  # Sử dụng set để loại bỏ nhãn trùng lặp
#         max_phrase_length = 3  # Giới hạn số từ trong cụm từ
#
#         for sentence_id, words in annotated_sentences.items():
#             if not isinstance(words, list):
#                 continue
#
#             for word_info in words:
#                 if not isinstance(word_info, dict):
#                     continue
#
#                 # Kiểm tra POS của từ hiện tại (chỉ lấy danh từ, danh từ riêng, tính từ)
#                 pos_tag = word_info.get('posTag', '')
#                 word_form = word_info.get('wordForm', '')
#                 if pos_tag not in ['N', 'Np', 'A']:
#                     continue
#
#                 # Annotate lại từ hiện tại để tách thêm các thành phần bên trong nếu có
#                 annotated_word = model.annotate_text(word_form)
#                 for sentence, word_a in annotated_word.items():
#                     if not isinstance(word_a, list):
#                         continue
#                     current_phrase = []  # Cụm từ hiện tại trong kết quả annotate lồng
#                     for word_if in word_a:
#                         if not isinstance(word_if, dict):
#                             continue
#                         if word_if['posTag'] == 'CH': continue
#                         word_form_if = word_if.get('wordForm', '')
#                         pos_tag_if = word_if.get('posTag', '')
#                         nerLabel = word_if.get('nerLabel', '')
#                         # Chỉ xử lý các từ có POS phù hợp
#                         if pos_tag_if == 'Np' and nerLabel == 'O' or nerLabel in ['B-LOC', 'B-PER']:
#                             current_phrase.append(word_form_if)
#                             generated_labels.add(word_form_if)  # Thêm từ đơn lẻ làm nhãn
#
#                     # Thêm cụm từ nếu cụm hiện tại không vượt quá giới hạn độ dài
#                     if current_phrase and len(current_phrase) <= max_phrase_length:
#                         phrase = " ".join(current_phrase)
#                         generated_labels.add(phrase)
#
#                 # Trích xuất thông tin từ annotate
#                 # word_form = word_info.get('wordForm', '')
#                 # pos_tag = word_info.get('posTag', '')
#                 #
#                 # if pos_tag in ['N', 'Np', 'A', 'Q']:  # Danh từ, danh từ riêng, tính từ, cảm thán
#                 #     current_phrase.append(word_form)
#                 #     generated_labels.add(word_form)  # Thêm từ đơn lẻ làm nhãn
#
#             # Thêm cụm từ hiện tại vào danh sách nếu không vượt quá giới hạn
#             # if current_phrase and len(current_phrase) <= max_phrase_length:
#             #     phrase = " ".join(current_phrase)
#             #     generated_labels.add(phrase)
#
#         # Chuyển set thành danh sách và sắp xếp nhãn
#         unique_labels = sorted(list(generated_labels))
#         logging.info(f"Nhãn được trích xuất: {unique_labels}")
#
#         # Gửi lại nhãn cho Spring
#         return jsonify({"labels": unique_labels})
#
#     except Exception as e:
#         logging.error(f"Lỗi khi xử lý yêu cầu: {str(e)}", exc_info=True)
#         return jsonify({"error": "Đã xảy ra lỗi trong quá trình xử lý"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
