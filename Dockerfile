# Sử dụng Python làm base image
FROM python:3.10

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy file ứng dụng và yêu cầu vào container
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
COPY clothing_index.faiss /app/clothing_index.faiss

# Cài đặt các thư viện cần thiết
RUN pip install -r requirements.txt

# Expose port để Flask có thể lắng nghe
EXPOSE 5000

# Command để chạy ứng dụng Flask
CMD ["python", "app.py"]
