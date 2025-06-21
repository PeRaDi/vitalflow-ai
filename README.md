# vitalflow-ai
Inventory Management System powered by AI

## Helpful commands
```
docker build -t ai-node .
docker run --rm --gpus all --env-file .env.trainer ai-node
docker run --rm --gpus all --env-file .env.forecaster ai-node
```