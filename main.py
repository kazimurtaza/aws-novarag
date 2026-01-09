"""
NovaRAG - Agentic RAG with Pydantic AI Framework and AWS Bedrock.

Uses Pydantic AI framework properly with:
- Agent class with dependency injection
- @agent.tool decorators for tools
- RunContext for accessing dependencies

Supports both Supabase and AWS RDS PostgreSQL with pgvector.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import psycopg2
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============
# Database Connection
# ==============

def get_db_connection():
    """Get database connection (Neon, RDS, or Supabase)."""
    # Option 1: DATABASE_URL (Neon, Railway, Render, etc.)
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        import psycopg2
        conn = psycopg2.connect(database_url, sslmode="require")
        logger.info("Connected via DATABASE_URL")
        return conn

    # Option 2: Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    if supabase_url and "your-project" not in supabase_url:
        from supabase import create_client
        return create_client(supabase_url, os.getenv("SUPABASE_KEY"))

    # Option 3: Individual connection params (RDS, etc.)
    db_host = os.getenv("DB_HOST")
    if db_host:
        import psycopg2
        conn = psycopg2.connect(
            host=db_host,
            database=os.getenv("DB_NAME", "novaragdb"),
            user=os.getenv("DB_USER", "novaragadmin"),
            password=os.getenv("DB_PASSWORD"),
            sslmode="require"
        )
        return conn

    logger.error("No database configured! Set DATABASE_URL or DB_HOST.")
    return None


db = get_db_connection()


@dataclass
class Deps:
    """Dependencies injected by Pydantic AI RunContext."""
    db: any  # Can be Supabase Client or psycopg2 connection


# ==============
# Bedrock Models
# ==============

class BedrockEmbeddingModel:
    """AWS Bedrock embedding model for Titan v2."""

    def __init__(self, region: str = "ap-southeast-2"):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_name = "amazon.titan-embed-text-v2:0"

    async def get_embedding(self, text: str) -> List[float]:
        """Get 1024-dim embedding for text."""
        request_body = {"inputText": text}
        response = self.client.invoke_model(
            modelId=self.model_name,
            body=json.dumps(request_body),
        )
        result = json.loads(response["body"].read())
        return result.get("embedding", [])


# Initialize models
aws_region = os.getenv("AWS_REGION", "ap-southeast-2")
chat_model = BedrockConverseModel('amazon.nova-lite-v1:0')
embedding_model = BedrockEmbeddingModel(region=aws_region)


# ==============
# Pydantic AI Agent
# ==============

SYSTEM_PROMPT = """
You are an expert at Pydantic AI - a Python AI agent framework.

You have access to comprehensive documentation through THREE RETRIEVAL STRATEGIES:

## Strategy 1: Semantic Search (retrieve_relevant_documentation)
Best for: Quick retrieval based on meaning, when you know relevant keywords
- Returns top 5 most relevant chunks based on vector similarity
- Includes metadata showing total chunks available per page
- Use this FIRST for most questions

## Strategy 2: Browse Documentation (list_documentation_pages)
Best for: When semantic search might miss relevant pages
- Lists ALL available documentation pages with titles
- Use when:
  * Semantic search returns poor/no results
  * You need to understand what topics are covered
  * Different terminology might be used than your search terms
  * User asks "what topics are covered?" or similar

## Strategy 3: Full Document View (get_page_content)
Best for: Getting complete context when chunks are incomplete
- Returns ALL chunks from a page in order
- Use when:
  * Retrieved chunks seem incomplete or fragmented
  * Content says "see more" or references other sections
  * You need full context to provide accurate answer
  * The page has many chunks and you need everything

## AGENTIC RAG APPROACH:
Traditional RAG is a "one-trick pony" - only semantic search. You are SMARTER.
You can CHOOSE the best strategy based on the question:

1. Most questions: Start with retrieve_relevant_documentation (fastest)
2. Poor results or terminology mismatch: Try list_documentation_pages to browse
3. Incomplete chunks: Use get_page_content for full context

THINK before answering: Which strategy best serves this question? Use multiple
strategies if needed. Be honest when documentation doesn't contain the answer.
"""


rag_agent = Agent(
    model=chat_model,
    deps_type=Deps,
    system_prompt=SYSTEM_PROMPT,
    retries=5,
)


def is_supabase(db) -> bool:
    """Check if database connection is Supabase."""
    return hasattr(db, 'table')


@rag_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[Deps],
    user_query: str,
    top_k: int = 5,
) -> str:
    """STRATEGY 1 - Semantic Search: Find relevant chunks by meaning.

    Use this FIRST for most questions. Fast retrieval based on vector similarity.

    When to use:
    - Most questions (start here)
    - You know the relevant keywords/concepts
    - Quick lookup needed

    Args:
        user_query: The search query to find relevant documentation
        top_k: Number of results to return (default: 5)

    Returns:
        Top K most relevant chunks with metadata showing total chunks per page.
        If a page has more chunks than retrieved, consider get_page_content().
    """
    query_embedding = await embedding_model.get_embedding(user_query)

    if is_supabase(ctx.deps.db):
        # Supabase path
        result = ctx.deps.db.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': top_k,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()

        if not result.data:
            return "No relevant documentation found."

        # Get total chunk counts for each URL
        urls = [doc['url'] for doc in result.data]
        chunk_counts = {}
        for url in urls:
            count_result = ctx.deps.db.table('site_pages') \
                .select('url') \
                .eq('url', url) \
                .execute()
            chunk_counts[url] = len(count_result.data)

        # Build formatted response with metadata
        formatted_chunks = []
        for i, doc in enumerate(result.data, 1):
            total_for_url = chunk_counts.get(doc['url'], 1)
            formatted_chunks.append(
                f"[Chunk {i}/{top_k} • Page has {total_for_url} total chunks]\n"
                f"# {doc['title']}\n"
                f"**URL:** {doc['url']}\n\n"
                f"{doc['content']}"
            )

        metadata = f"\n📊 **Retrieved {len(result.data)} chunks** from {len(set(urls))} page(s)."
        if any(chunk_counts[url] > 1 for url in urls):
            for url in set(urls):
                if chunk_counts[url] > 1:
                    metadata += f"\n   • {url}: {chunk_counts[url]} chunks total"
            metadata += f"\n\n**IMPORTANT:** If you need more context, use get_page_content() with ONE of these EXACT URLs:"
            for url in set(urls):
                metadata += f"\n   • get_page_content(url=\"{url}\")"

        return metadata + "\n\n" + "\n\n---\n\n".join(formatted_chunks)
    else:
        # RDS PostgreSQL path - convert list to vector string for PostgreSQL
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        cursor = ctx.deps.db.cursor()
        cursor.execute(
            "SELECT url, title, content FROM match_site_pages(%s::vector, %s, %s)",
            (embedding_str, top_k, json.dumps({'source': 'pydantic_ai_docs'}))
        )
        results = cursor.fetchall()

        if not results:
            return "No relevant documentation found."

        # Get total chunk counts for each URL
        urls = [row[0] for row in results]
        chunk_counts = {}
        for url in urls:
            cursor.execute(
                "SELECT COUNT(*) FROM site_pages WHERE url = %s AND metadata->>'source' = 'pydantic_ai_docs'",
                (url,)
            )
            chunk_counts[url] = cursor.fetchone()[0]

        # Build formatted response with metadata
        formatted_chunks = []
        for i, row in enumerate(results, 1):
            url, title, content = row
            total_for_url = chunk_counts.get(url, 1)
            formatted_chunks.append(
                f"[Chunk {i}/{top_k} • Page has {total_for_url} total chunks]\n"
                f"# {title}\n"
                f"**URL:** {url}\n\n"
                f"{content}"
            )

        metadata = f"\n📊 **Retrieved {len(results)} chunks** from {len(set(urls))} page(s)."
        if any(chunk_counts[url] > 1 for url in urls):
            for url in set(urls):
                if chunk_counts[url] > 1:
                    metadata += f"\n   • {url}: {chunk_counts[url]} chunks total"
            metadata += f"\n\n**IMPORTANT:** If you need more context, use get_page_content() with ONE of these EXACT URLs:"
            for url in set(urls):
                metadata += f"\n   • get_page_content(url=\"{url}\")"

        return metadata + "\n\n" + "\n\n---\n\n".join(formatted_chunks)


@rag_agent.tool
async def list_documentation_pages(
    ctx: RunContext[Deps],
    limit: int = 200,
) -> str:
    """Browse ALL available Pydantic AI documentation pages (sitemap/table of contents).

    STRATEGY 2 - Use this when:
    - Semantic search (retrieve_relevant_documentation) returns poor/no results
    - The question might use different terminology than the documentation
    - You need to understand what topics are covered
    - User asks "what topics/pages are available?"

    Returns a list of all pages with their URLs and titles, grouped by category.
    This lets you browse the entire documentation structure like a table of contents.

    Args:
        limit: Maximum number of pages to return (default: 200)

    Returns:
        Formatted list of pages with URLs and titles
    """
    if is_supabase(ctx.deps.db):
        result = ctx.deps.db.table('site_pages') \
            .select('url, title') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .limit(limit * 2) \
            .execute()

        if not result.data:
            return "No documentation pages found."

        # Get unique pages with titles (first chunk's title)
        pages = {}
        for doc in result.data:
            url = doc['url']
            if url not in pages:
                # Extract the main title (remove chunk suffix if present)
                title = doc['title'].split(' - ')[0] if ' - ' in doc['title'] else doc['title']
                pages[url] = title

        # Sort by URL and format
        formatted = ["## Available Pydantic AI Documentation Pages\n"]
        formatted.append(f"Total pages: {len(pages)}\n")

        # Group by URL prefix for better organization
        groups = {}
        for url, title in sorted(pages.items()):
            prefix = url.split(':')[0] if ':' in url else 'other'
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((url, title))

        for prefix, pages_list in sorted(groups.items()):
            formatted.append(f"\n### {prefix.upper()}")
            for url, title in pages_list[:20]:  # Limit per group
                formatted.append(f"- {title}")
                formatted.append(f"  URL: `{url}`")
            if len(pages_list) > 20:
                formatted.append(f"  ... and {len(pages_list) - 20} more")

        return "\n".join(formatted)

    else:
        cursor = ctx.deps.db.cursor()
        cursor.execute(
            "SELECT DISTINCT url, title FROM site_pages WHERE metadata->>'source' = 'pydantic_ai_docs' LIMIT %s",
            (limit * 2,)
        )
        results = cursor.fetchall()

        if not results:
            return "No documentation pages found."

        # Get unique pages with titles
        pages = {}
        for url, title in results:
            if url not in pages:
                title = title.split(' - ')[0] if ' - ' in title else title
                pages[url] = title

        # Sort by URL and format
        formatted = ["## Available Pydantic AI Documentation Pages\n"]
        formatted.append(f"Total pages: {len(pages)}\n")

        # Group by URL prefix
        groups = {}
        for url, title in sorted(pages.items()):
            prefix = url.split(':')[0] if ':' in url else 'other'
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((url, title))

        for prefix, pages_list in sorted(groups.items()):
            formatted.append(f"\n### {prefix.upper()}")
            for url, title in pages_list[:20]:
                formatted.append(f"- {title}")
                formatted.append(f"  URL: `{url}`")
            if len(pages_list) > 20:
                formatted.append(f"  ... and {len(pages_list) - 20} more")

        return "\n".join(formatted)


@rag_agent.tool
async def get_page_content(
    ctx: RunContext[Deps],
    url: str,
) -> str:
    """STRATEGY 3 - Full Document View: Get complete page content.

    Use when semantic search returns incomplete chunks. This retrieves ALL
    chunks from a page in order, not just the most relevant ones.

    When to use:
    - Retrieved chunks seem incomplete or fragmented
    - Content says "see more" or references other sections
    - You need full context for accurate answer
    - Page has many chunks and you need everything

    Args:
        url: The exact URL from list_documentation_pages or retrieve_relevant_documentation

    Returns:
        Full page content with ALL chunks combined in order
    """
    if is_supabase(ctx.deps.db):
        result = ctx.deps.db.table('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()

        if not result.data:
            # Try to suggest similar URLs
            cursor = ctx.deps.db.cursor()
            cursor.execute(
                "SELECT DISTINCT url FROM site_pages WHERE metadata->>'source' = 'pydantic_ai_docs' LIMIT 20"
            )
            available_urls = [row[0] for row in cursor.fetchall()]
            return f"ERROR: URL '{url}' not found. Available URLs start with: {', '.join(set([u.split(':')[0] + ':' for u in available_urls[:5]]))}... Use get_page_content() with an EXACT URL from retrieve_relevant_documentation results."

        page_title = result.data[0]['title'].split(' - ')[0]
        formatted = [f"# {page_title}\n"]

        for chunk in result.data:
            formatted.append(chunk['content'])

        return "\n\n".join(formatted)
    else:
        cursor = ctx.deps.db.cursor()
        cursor.execute(
            "SELECT title, content FROM site_pages WHERE url = %s AND metadata->>'source' = 'pydantic_ai_docs' ORDER BY chunk_number",
            (url,)
        )
        results = cursor.fetchall()

        if not results:
            # Suggest available URLs
            cursor.execute(
                "SELECT DISTINCT url FROM site_pages WHERE metadata->>'source' = 'pydantic_ai_docs' LIMIT 20"
            )
            available_urls = [row[0] for row in cursor.fetchall()]
            return f"ERROR: URL '{url}' not found. Available URLs start with: {', '.join(set([u.split(':')[0] + ':' for u in available_urls[:5]]))}... Use get_page_content() with an EXACT URL from retrieve_relevant_documentation results."

        page_title = results[0][0].split(' - ')[0]
        formatted = [f"# {page_title}\n"]

        for row in results:
            formatted.append(row[1])

        return "\n\n".join(formatted)


# ==============
# Metrics Tracking
# ==============

class MetricsTracker:
    """Track query metrics."""
    PRICING = {
        "amazon.nova-lite-v1:0": {"input": 0.15, "output": 0.60},
        "amazon.titan-embed-text-v2:0": {"input": 0.02, "output": 0.0},
    }

    def __init__(self):
        self.start_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
        self.tool_calls = []

    def start(self):
        import time
        self.start_time = time.time()

    def get_summary(self) -> dict:
        import time
        elapsed_ms = int((time.time() - self.start_time) * 1000) if self.start_time else 0

        chat_cost = (
            self.input_tokens * self.PRICING["amazon.nova-lite-v1:0"]["input"] / 1_000_000 +
            self.output_tokens * self.PRICING["amazon.nova-lite-v1:0"]["output"] / 1_000_000
        )
        embedding_cost = self.embedding_tokens * self.PRICING["amazon.titan-embed-text-v2:0"]["input"] / 1_000_000

        return {
            "latency_ms": elapsed_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.input_tokens + self.output_tokens + self.embedding_tokens,
            "estimated_cost_usd": round(chat_cost + embedding_cost, 6),
            "tool_calls": self.tool_calls,
        }


metrics_tracker = MetricsTracker()


# ==============
# Main CLI
# ==============

async def main():
    """Main entry point for CLI usage."""
    print("=" * 70)
    print("NovaRAG - Pydantic AI + AWS Bedrock + RDS/Supabase")
    print("=" * 70)
    print()

    if not db:
        print("ERROR: Database not configured!")
        print()
        print("Please configure one of the following:")
        print()
        print("Option 1: Supabase")
        print("  1. Create project at https://supabase.com")
        print("  2. Enable 'vector' extension")
        print("  3. Run scripts/supabase_setup.sql")
        print("  4. Update .env with SUPABASE_URL and SUPABASE_KEY")
        print()
        print("Option 2: AWS RDS PostgreSQL")
        print("  1. Run: bash scripts/deploy_rds.sh")
        print("  2. Update .env with DB_HOST, DB_NAME, DB_USER, DB_PASSWORD")
        print("  3. Run: psql -h <endpoint> -U <user> -d <db> -f scripts/rds_pgvector_setup.sql")
        print()
        return

    db_type = "Supabase" if is_supabase(db) else "AWS RDS PostgreSQL"
    print(f"Connected to: {db_type}")
    print(f"Model: amazon.nova-lite-v1:0")
    print(f"Region: {aws_region}")
    print()
    print("Type 'quit' to exit")
    print("-" * 70)

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nProcessing...")

            # Track metrics
            metrics_tracker.start()
            metrics_tracker.embedding_tokens = len(question.split()) * 1.3  # Rough estimate

            deps = Deps(db=db)
            result = await rag_agent.run(question, deps=deps)

            # Update token usage
            usage = result.usage()
            metrics_tracker.input_tokens = usage.input_tokens
            metrics_tracker.output_tokens = usage.output_tokens

            summary = metrics_tracker.get_summary()

            print("\n" + "=" * 70)
            print("ANSWER:")
            print("=" * 70)
            print(result.output)

            print()
            print("Metrics:")
            print(f"  Latency: {summary['latency_ms']}ms")
            print(f"  Tokens: {summary['total_tokens']} ({summary['input_tokens']} in, {summary['output_tokens']} out)")
            print(f"  Embedding tokens: {int(summary['embedding_tokens'])}")
            print(f"  Cost: ${summary['estimated_cost_usd']:.6f}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception(e)


if __name__ == "__main__":
    asyncio.run(main())
