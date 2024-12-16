from faiss import IndexIDMap
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Cấu hình kết nối
db_config = URL.create(
    "mysql+pymysql",
    "root",
    "123456aA@",
    "localhost",
    3306,
    "ecommerce"
)

try:
    # Tạo engine kết nối
    engine = create_engine(db_config)

    # Truy vấn dữ liệu từ bảng
    query = "SELECT id, score, create_at, update_at, price, signature, title, description FROM products"
    data = pd.read_sql(query, con=engine)

    print("Dữ liệu đã được tải thành công!")

    # Kiểm tra dữ liệu
    if data.empty:
        raise ValueError("Không có dữ liệu nào được truy vấn từ bảng products.")

    # Tải mô hình nhúng (embedding model)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Tạo vector từ cột 'description'
    print("Đang vector hóa dữ liệu mô tả sản phẩm...")
    data['vector'] = data['title'].apply(lambda x: model.encode(x).tolist())
    print("Đang lưu vector")
    data.to_json("vectorized_data.json", orient='records')
    if data.empty:
        raise ValueError("Không có dữ liệu nào được tạo vector.")
    else:
        print("Tạo vector thành công")

    # Chuẩn bị vector và ID sản phẩm
    print("Chuẩn bị vector và ID sản phẩm")
    vectors = np.array(data['vector'].tolist(), dtype='float32')
    ids = np.array(data['id'].tolist(), dtype='int64')

    # Tạo FAISS index
    print("Đang tạo FAISS index...")
    index = IndexIDMap(faiss.IndexFlatL2(vectors.shape[1]))  # Sử dụng khoảng cách L2
    index.add_with_ids(vectors, np.array(ids, dtype='int64'))

    # Lưu chỉ mục FAISS
    faiss_file = "clothing_index.faiss"
    faiss.write_index(index, faiss_file)
    print(f"FAISS index đã được lưu vào file: {faiss_file}")

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")

finally:
    # Đảm bảo đóng kết nối nếu không dùng engine
    engine.dispose()
    print("Đóng kết nối thành công.")
