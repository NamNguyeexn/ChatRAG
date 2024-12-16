import requests
from openai import OpenAI
from faiss import IndexIDMap
from flask import Flask, request, jsonify
import faiss
import numpy as np
import pandas as pd
from sqlalchemy import URL, create_engine

app = Flask(__name__)

# OpenAI Configuration
token = "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"
model_name = "text-embedding-ada-002"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Load FAISS index
print("Đang tải FAISS index...")
index = faiss.read_index("clothing_index.faiss") # Tệp FAISS đã được tạo trước

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
    Vector hóa dữ liệu từ mô tả sản phẩm bằng text-embedding-ada-002.
    """
    try:
        embeddings = []
        for product in data.itertuples():
            text_to_vectorize = f"{product.title} {product.description}"
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json={"input": text_to_vectorize, "model": model_name}
            )
            response.raise_for_status()
            embeddings.append(response.json()["data"][0]["embedding"])
        data['vector'] = embeddings
        return data
    except Exception as e:
        raise ValueError(f"Lỗi khi vector hóa dữ liệu: {e}")

def create_faiss_index(data, faiss_file):
    """
    Tạo FAISS index và lưu vào file.
    """
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

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        # Nhận câu hỏi từ request
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({"error": "Câu hỏi không được để trống"}), 400

        # Tạo embedding từ câu hỏi bằng text-embedding-ada-002
        print(f"Nhận câu hỏi: {question}")
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json={"input": question, "model": model_name}
        )
        response.raise_for_status()
        question_embedding = np.array(response.json()["data"][0]["embedding"], dtype='float32').reshape(1, -1)

        # Tìm kiếm trong FAISS
        print("Tìm kiếm trong FAISS...")
        distances, indices = index.search(question_embedding, k=5)  # Tìm câu trả lời gần nhất (k=5)

        # Lấy câu trả lời từ danh sách
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(answers):
                results.append({
                    "answer": answers[idx],
                    "distance": float(distances[0][i])
                })

        # Trả kết quả về cho Flask
        return jsonify({
            "question": question,
            "results": results
        })
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
