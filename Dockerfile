FROM python:3.12-slim

WORKDIR /app

# 1. Copiar requirements e instalar dependencias primero
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# 2. Copiar el resto del código
COPY . .

# 3. Exponer el puerto
EXPOSE 8000

# 4. Ejecutar servidor Uvicorn apuntando a la api
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
