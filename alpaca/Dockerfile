FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV HF_HOME=/app/.cache

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD streamlit run --server.port 80 app.py
