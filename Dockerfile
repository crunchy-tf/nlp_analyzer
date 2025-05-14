# Dockerfile content
FROM python:3.10-slim
WORKDIR /service
COPY ./requirements.txt /service/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /service/requirements.txt
COPY ./app /service/app
COPY ./bertopic_model_final_guided_multilang_gensim.joblib /service/bertopic_model_final_guided_multilang_gensim.joblib
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]