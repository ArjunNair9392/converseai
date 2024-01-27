FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
COPY upload.py ./uploadApp.py
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "uploadApp.py"]