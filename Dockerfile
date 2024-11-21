# Menggunakan image dasar Python 3.9-slim
FROM python:3.9-slim

# Menetapkan variabel lingkungan untuk menghindari buffer
ENV PYTHONUNBUFFERED=1

# Mengatur direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt ke direktori kerja
COPY requirements.txt .

# Menginstall dependensi Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode aplikasi ke direktori kerja
COPY . .

# Menentukan port yang akan diekspos
EXPOSE 5000

# Menggunakan Gunicorn sebagai WSGI server untuk menjalankan aplikasi Flask
CMD ["gunicorn", "run:app", "--bind", "0.0.0.0:5000", "--workers", "4"]

