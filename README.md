# vitalflow-ai
Inventory Management System powered by AI

## Helpful commands
```
docker build -t ai-node .
docker run --rm --gpus all -e NODE_TYPE=trainer ai-node
docker run --rm --gpus all -e NODE_TYPE=forecaster ai-node
```