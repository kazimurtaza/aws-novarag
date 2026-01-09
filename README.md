# aws-novarag

Agentic RAG using AWS Bedrock (Nova + Titan) and Pydantic AI framework.

## What is it?

A smarter RAG that chooses retrieval strategies instead of just semantic search:
1. **Semantic Search** - Vector similarity (default)
2. **Browse Pages** - List all available docs
3. **Full Page View** - Get complete document context

## Stack

- **LLM**: AWS Bedrock Nova Lite
- **Embeddings**: AWS Titan Embeddings v2
- **Framework**: Pydantic AI
- **Database**: PostgreSQL + pgvector (RDS or Supabase)

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/aws-novarag.git
cd aws-novarag
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your AWS credentials and database config

# 3. Ingest data
python scripts/step1_generate_embeddings.py
python scripts/step2_upsert_to_rds.py

# 4. Run
python main.py
```

## API Server

```bash
cd deployment && python app.py
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Ask a question |
| `/health` | GET | Health check |
| `/stats` | GET | Query statistics |

## Deploy to AWS

```bash
./deploy.sh
```

Deploys: RDS PostgreSQL + ECS Fargate

## Cost

| Model | Price |
|-------|-------|
| Nova Lite | $0.15/M input, $0.60/M output |
| Titan Embeddings | $0.02/M tokens |

**~$0.0005 per query** (~$1.50/month at 100 queries/day)

## License

MIT
