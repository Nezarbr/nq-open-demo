# NQ-open retrieval demo — CPU-only image.
# Models are baked in at build time, so a pod never downloads them at start; the 6 GB data folder is NOT in the
# image: it is mounted read-only from the Runpod network volume at /workspace/demo.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

RUN useradd --create-home --uid 1000 app
WORKDIR /srv

COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the two public models (retriever query encoder + our fine-tuned reranker)
RUN python -c "\
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('facebook/contriever-msmarco'); AutoModel.from_pretrained('facebook/contriever-msmarco'); \
AutoTokenizer.from_pretrained('NezarIn/nq-open-minilm-reranker'); \
AutoModelForSequenceClassification.from_pretrained('NezarIn/nq-open-minilm-reranker', num_labels=1)" \
 && chown -R app /opt/hf

COPY app/ ./app/

# no network for models at runtime; run as a non-root user
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
