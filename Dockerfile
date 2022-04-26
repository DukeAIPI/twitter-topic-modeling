FROM amazon/aws-lambda-python:3.8

RUN mkdir -p /app
COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD [ "streamlit.py" ]]
ENTRYPOINT [ "python" ]