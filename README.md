# vitalflow-ai
Inventory Management System powered by AI

## Helpful commands
```
docker build -t ai-node .
docker run --rm --gpus all -e \
    TASK_TYPE=trainer \
    RABBITMQ_HOST=vital-flow.live \
    RABBITMQ_PORT=5672 \
    RABBITMQ_USERNAME= \
    RABBITMQ_PASSWORD= \
    DATABASE_HOST=vital-flow.live \
    DATABASE_PORT=5432 \
    DATABASE_NAME= \
    DATABASE_USERNAME= \
    DATABASE_PASSWORD= \
    CDN_HOST=cdn.vital-flow.live \
    CDN_MODELS_PATH=models \
    CDN_PORT=88 \
    CDN_USERNAME= \
    CDN_PASSWORD= \
    ai-node

docker run --rm --gpus all -e \
    TASK_TYPE=forecaster \
    RABBITMQ_HOST=vital-flow.live \
    RABBITMQ_PORT=5672 \
    RABBITMQ_USERNAME= \
    RABBITMQ_PASSWORD= \
    DATABASE_HOST=vital-flow.live \
    DATABASE_PORT=5432 \
    DATABASE_NAME= \
    DATABASE_USERNAME= \
    DATABASE_PASSWORD= \
    CDN_HOST=cdn.vital-flow.live \
    CDN_MODELS_PATH=models \
    CDN_PORT=88 \
    CDN_USERNAME= \
    CDN_PASSWORD= \
    ai-node
```