FROM python:3.8-slim

WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["streamlit.py"]
