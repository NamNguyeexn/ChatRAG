from datetime import datetime
import faiss
import numpy as np
import pandas as pd
from faiss import IndexIDMap
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sqlalchemy import URL, create_engine
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from underthesea import word_tokenize, pos_tag

app = Flask(__name__)

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
index = faiss.read_index("clothing_index.faiss")  # Tệp FAISS đã được tạo trước

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

FAISS_FILE = "clothing_index.faiss"


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
        return ' '.join(keywords)
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

    #Su dung gpt de lay tu khoa
        # gpt_response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {"role": "system", "content": "Bạn là một trợ lý AI giúp trích xuất từ khóa và phân tích câu hỏi."},
        #         {"role": "user", "content": f"Trích xuất các từ khóa chính từ câu hỏi sau: '{question}'"}
        #     ]
        # )

        # Truy cập kết quả phản hồi
        # extracted_keywords = gpt_response.choices[0].message.content
        extracted_keywords = extract_keywords_vietnamese(question)
        print(f"Từ khóa được trích xuất: {extracted_keywords}")

        # Tạo embedding từ từ khóa được trích xuất
        question_embedding = embedding_model.encode(extracted_keywords).astype('float32').reshape(1, -1)

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
        # for i, product in products.iterrows():
        #     product_info = f"Tên: {product['title']}, Giá: {product['price']}, Mô tả: {product['description']}"
        #     gpt_response = client.chat.completions.create(
        #         model=model_name,
        #         messages=[
        #             {"role": "system", "content": "Bạn là một trợ lý AI giúp tạo câu trả lời từ thông tin sản phẩm."},
        #             {"role": "user", "content": f"Tạo câu trả lời từ thông tin sản phẩm: '{product_info}'"}
        #         ]
        #     )
        #     answer = gpt_response.choices[0].message.content
        #     results.append(product['id']
        #                    # "answer": answer,
        #                    # "distance": float(distances[0][i])
        #                    )

        return jsonify({
            "question": question,
            "results": results
        })
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
