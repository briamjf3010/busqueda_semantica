# Usar una imagen ligera de Python
FROM python:3.9-slim

# Instalar nano u otras dependencias del sistema
RUN apt-get update && apt-get install -y nano

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requisitos primero para instalar las dependencias
COPY requirements.txt .

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos a la imagen del contenedor
COPY src/ ./src/

# Comando que ejecutará el script principal
CMD ["python", "src/main.py"]