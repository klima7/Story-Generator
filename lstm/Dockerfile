FROM bitnami/pytorch:2.0.1

USER root

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 80

CMD streamlit run --server.port 80 app.py
