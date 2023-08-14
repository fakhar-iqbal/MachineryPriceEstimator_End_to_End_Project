FROM python:3.10.12-slim
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

CMD [ "python3","application.py" ]
