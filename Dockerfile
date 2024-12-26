# Sử dụng Python làm base image
FROM python:3.10

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy file ứng dụng và yêu cầu vào container
COPY app.py /app/app.py
#COPY requirements.txt /app/requirements.txt
COPY clothing_index.faiss /app/clothing_index.faiss

# Cài đặt các thư viện cần thiết
RUN #pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Expose port để Flask có thể lắng nghe
EXPOSE 5000

# Thiết lập biến môi trường để Flask chạy trên 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Thiết lập các biến môi trường MySQL
ENV MYSQL_HOST=localhost
ENV MYSQL_PORT=3306
ENV MYSQL_DATABASE=ecommerce

# Command để chạy ứng dụng Flask
CMD ["python", "app.py"]
