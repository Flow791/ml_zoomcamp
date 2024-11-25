FROM python:3.11-slim

RUN pip install pipenv

WORKDIR /app
COPY ["env/Pipfile", "env/Pipfile.lock", "model/model_result_home.bin", "model/model_result_away.bin", "./"]

RUN pipenv lock
RUN pipenv install --system --deploy

COPY ["python/web_service.py", "./"]

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "web_service:app"]