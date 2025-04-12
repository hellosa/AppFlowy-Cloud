import argparse
import asyncio
import json
import logging
import os
import ssl # Import ssl module
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp
import asyncpg
from dotenv import load_dotenv
from pydantic import BaseModel

# --- Configuration & Argument Parsing ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("batch_indexer")


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


class OpenAIConfig(BaseModel):
    api_key: str
    api_base: Optional[str] = None
    model: str


class AzureConfig(BaseModel):
    api_key: str
    endpoint: str
    deployment_id: str


@dataclass
class Args:
    database_url: str
    openai_config: Optional[OpenAIConfig] = None
    azure_config: Optional[AzureConfig] = None
    max_concurrency: int = 4 # Max parallel embedding tasks
    workspace_id: Optional[str] = None
    limit: Optional[int] = None
    fetch_batch_size: int = 100 # How many doc IDs to fetch at once
    collab_type_document: int = 0 # From previous search


def parse_args() -> Args:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AppFlowy Batch Indexer")

    parser.add_argument("--database-url", default=get_env_var("DATABASE_URL"), required=get_env_var("DATABASE_URL") is None, help="PostgreSQL connection URL (e.g., postgresql://user:pass@host/db)")
    parser.add_argument("--max-concurrency", type=int, default=get_env_var("BACKGROUND_INDEXER_THREADS", 4), help="Maximum concurrent embedding tasks")
    parser.add_argument("--workspace-id", default=get_env_var("WORKSPACE_ID"), help="Optional: Process only a specific workspace UUID")
    parser.add_argument("--limit", type=int, default=get_env_var("BATCH_LIMIT"), help="Optional: Limit the total number of documents to process")
    parser.add_argument("--fetch-batch-size", type=int, default=get_env_var("FETCH_BATCH_SIZE", 100), help="Number of document IDs to fetch from DB at a time")

    # OpenAI Args
    parser.add_argument("--openai-api-key", default=get_env_var("OPENAI_API_KEY"))
    parser.add_argument("--openai-api-base", default=get_env_var("OPENAI_API_BASE"))
    parser.add_argument("--openai-model", default=get_env_var("OPENAI_MODEL"))

    # Azure Args
    parser.add_argument("--azure-api-key", default=get_env_var("AZURE_AI_API_KEY"))
    parser.add_argument("--azure-endpoint", default=get_env_var("AZURE_AI_ENDPOINT"))
    parser.add_argument("--azure-deployment-id", default=get_env_var("AZURE_AI_DEPLOYMENT_ID"))

    parsed = parser.parse_args()

    openai_cfg = None
    if parsed.openai_api_key and parsed.openai_model:
        openai_cfg = OpenAIConfig(
            api_key=parsed.openai_api_key,
            api_base=parsed.openai_api_base,
            model=parsed.openai_model,
        )

    azure_cfg = None
    if parsed.azure_api_key and parsed.azure_endpoint and parsed.azure_deployment_id:
        azure_cfg = AzureConfig(
            api_key=parsed.azure_api_key,
            endpoint=parsed.azure_endpoint,
            deployment_id=parsed.azure_deployment_id,
        )

    if not openai_cfg and not azure_cfg:
        parser.error("Either OpenAI or Azure AI configuration must be provided.")

    return Args(
        database_url=parsed.database_url,
        openai_config=openai_cfg,
        azure_config=azure_cfg,
        max_concurrency=parsed.max_concurrency,
        workspace_id=parsed.workspace_id,
        limit=parsed.limit,
        fetch_batch_size=parsed.fetch_batch_size,
    )

# --- Data Structures (Simplified) ---

@dataclass
class DocInfo:
    object_id: uuid.UUID
    workspace_id: uuid.UUID

@dataclass
class Chunk:
    fragment_id: str
    object_id: uuid.UUID
    content: Optional[str]
    embedding: Optional[List[float]] = None
    chunk_index: int = 0
    paragraph_index: int = 0

@dataclass
class EmbeddingResult:
    tokens_consumed: int
    chunks: List[Chunk]

@dataclass
class EmbeddingRecord:
    workspace_id: uuid.UUID
    object_id: uuid.UUID
    collab_type: str # Store as string '0' for simplicity
    tokens_used: int
    chunks: List[Chunk]


# --- Embedder Logic (Copied from previous version) ---

class Embedder:
    def __init__(self, args: Args):
        self.args = args
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        # WARNING: Disabling SSL verification is insecure. Only use if you trust the endpoint.
        # Create an SSL context that does not verify certificates
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        logger.warning("Disabling SSL certificate verification for HTTP client. Ensure you trust the API endpoint(s).")

        # Create a TCPConnector with the custom SSL context
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        # Create the ClientSession with the custom connector
        self.session = aiohttp.ClientSession(connector=connector)

    async def stop(self):
        if self.session:
            await self.session.close()

    def model_name(self) -> str:
        if self.args.openai_config:
            return self.args.openai_config.model
        if self.args.azure_config:
            return self.args.azure_config.deployment_id
        return "unknown"

    async def embed(self, chunks_to_embed: List[Chunk]) -> Optional[EmbeddingResult]:
        if not chunks_to_embed:
            return EmbeddingResult(tokens_consumed=0, chunks=[])

        texts = [chunk.content for chunk in chunks_to_embed if chunk.content is not None]
        if not texts:
            return EmbeddingResult(tokens_consumed=0, chunks=chunks_to_embed)

        if self.args.openai_config:
            return await self._embed_openai(texts, chunks_to_embed)
        elif self.args.azure_config:
            return await self._embed_azure(texts, chunks_to_embed)
        else:
            logger.error("No embedder configured")
            return None

    async def _embed_openai(self, texts: List[str], original_chunks: List[Chunk]) -> Optional[EmbeddingResult]:
        config = self.args.openai_config
        if not config or not self.session:
            return None
        url = (config.api_base or "https://api.openai.com/v1") + "/embeddings"
        headers = {"Authorization": f"Bearer {config.api_key}"}
        payload = {"input": texts, "model": config.model}
        try:
            async with self.session.post(url, json=payload, headers=headers, timeout=60) as response:
                response.raise_for_status()
                result = await response.json()
                embeddings = [item['embedding'] for item in result['data']]
                tokens = result.get('usage', {}).get('total_tokens', 0)
                embedded_chunk_idx = 0
                final_chunks = []
                for chunk in original_chunks:
                    if chunk.content is not None:
                        if embedded_chunk_idx < len(embeddings):
                           chunk.embedding = embeddings[embedded_chunk_idx]
                           embedded_chunk_idx += 1
                        else:
                            logger.error(f"Mismatch embedding results for chunk {chunk.fragment_id}")
                    final_chunks.append(chunk)
                return EmbeddingResult(tokens_consumed=tokens, chunks=final_chunks)
        except asyncio.TimeoutError:
             logger.error(f"OpenAI API request timed out after 60 seconds for {len(texts)} texts.")
             return None
        except aiohttp.ClientResponseError as e:
            logger.error(f"OpenAI API HTTP error: {e.status} {e.message}. URL: {url}")
            try:
                error_body = await e.text()
                logger.error(f"OpenAI Error Body: {error_body}")
            except Exception:
                pass # Ignore if error body cannot be read
            return None
        except Exception as e:
            logger.exception(f"Error processing OpenAI response: {e}")
            return None

    async def _embed_azure(self, texts: List[str], original_chunks: List[Chunk]) -> Optional[EmbeddingResult]:
        config = self.args.azure_config
        if not config or not self.session:
            return None
        url = f"{config.endpoint}/openai/deployments/{config.deployment_id}/embeddings?api-version=2023-05-15"
        headers = {"api-key": config.api_key}
        payload = {"input": texts}
        try:
            async with self.session.post(url, json=payload, headers=headers, timeout=60) as response:
                response.raise_for_status()
                result = await response.json()
                embeddings = [item['embedding'] for item in result['data']]
                tokens = result.get('usage', {}).get('total_tokens', 0)
                embedded_chunk_idx = 0
                final_chunks = []
                for chunk in original_chunks:
                    if chunk.content is not None:
                        if embedded_chunk_idx < len(embeddings):
                           chunk.embedding = embeddings[embedded_chunk_idx]
                           embedded_chunk_idx += 1
                        else:
                             logger.error(f"Mismatch embedding results for chunk {chunk.fragment_id}")
                    final_chunks.append(chunk)
                return EmbeddingResult(tokens_consumed=tokens, chunks=final_chunks)
        except asyncio.TimeoutError:
             logger.error(f"Azure API request timed out after 60 seconds for {len(texts)} texts.")
             return None
        except aiohttp.ClientResponseError as e:
            logger.error(f"Azure API HTTP error: {e.status} {e.message}. URL: {url}")
            try:
                error_body = await e.text()
                logger.error(f"Azure Error Body: {error_body}")
            except Exception:
                pass
            return None
        except Exception as e:
            logger.exception(f"Error processing Azure response: {e}")
            return None

# --- Placeholder Content Fetching ---
async def get_document_paragraphs_placeholder(pool: asyncpg.Pool, object_id: uuid.UUID) -> Optional[List[str]]:
    """ ** THIS IS A PLACEHOLDER **
        Replace this function with logic to fetch the actual content
        for the given object_id from your database/storage and return
        it as a list of strings (paragraphs).

        This might involve fetching a blob, decoding it using Rust logic
        (via FFI or a separate service), or querying a pre-processed text table.
    """
    logger.debug(f"Using placeholder content for {object_id}")
    # Example: Simulate fetching some generic content
    await asyncio.sleep(0.01) # Simulate tiny DB delay
    return [
        f"This is placeholder paragraph 1 for document {object_id}.",
        "It does not represent the real content.",
        "You MUST replace the get_document_paragraphs_placeholder function.",
        "Otherwise, incorrect embeddings will be stored."
    ]

# --- Chunking Logic (Simplified - Copied) ---

def create_chunks(object_id: uuid.UUID, paragraphs: List[str], model: str) -> List[Chunk]:
    chunks = []
    for i, para in enumerate(paragraphs):
        if para.strip():
            fragment_id = f"{i}:0"
            chunks.append(Chunk(
                fragment_id=fragment_id,
                object_id=object_id,
                content=para,
                paragraph_index=i,
                chunk_index=0
            ))
    return chunks

# --- Database Operations (Partially Copied/Adapted) ---

async def get_collab_embedding_fragment_ids(pool: asyncpg.Pool, object_id: uuid.UUID) -> set[str]:
    query = """
    SELECT fragment_id FROM af_collab_embeddings
    WHERE oid = $1
    """
    try:
        rows = await pool.fetch(query, object_id)
        return {row['fragment_id'] for row in rows}
    except Exception as e:
        logger.error(f"Failed to get collab embedding fragment ids for {object_id}: {e}")
        return set()

async def write_embeddings_to_db(pool: asyncpg.Pool, record: EmbeddingRecord):
    async with pool.acquire() as conn:
        async with conn.transaction():
            try:
                existing_query = "SELECT fragment_id FROM af_collab_embeddings WHERE oid = $1"
                existing_rows = await conn.fetch(existing_query, record.object_id)
                existing_fragment_ids = {row['fragment_id'] for row in existing_rows}
                new_fragment_ids = {chunk.fragment_id for chunk in record.chunks}
                to_delete = existing_fragment_ids - new_fragment_ids

                if to_delete:
                    delete_query = "DELETE FROM af_collab_embeddings WHERE oid = $1 AND fragment_id = ANY($2::varchar[])"
                    await conn.execute(delete_query, record.object_id, list(to_delete))
                    logger.debug(f"Deleted {len(to_delete)} old chunks for {record.object_id}")

                chunks_to_write = [chunk for chunk in record.chunks if chunk.embedding is not None]
                if chunks_to_write:
                    upsert_query = """
                    INSERT INTO af_collab_embeddings (oid, fragment_id, content_type, content, embedding, metadata, fragment_index, embedder_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (oid, fragment_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        fragment_index = EXCLUDED.fragment_index,
                        embedder_type = EXCLUDED.embedder_type,
                        indexed_at = NOW();
                    """
                    data_to_insert = [
                        (
                            chunk.object_id,
                            chunk.fragment_id,
                            0, # content_type (assuming 0 = PlainText)
                            chunk.content,
                            # Convert list to pgvector string format '[1.0,2.0,...]'
                            '[' + ",".join(map(str, chunk.embedding)) + ']' if chunk.embedding else None,
                            json.dumps({}), # metadata (empty JSON object)
                            chunk.paragraph_index, # Use paragraph_index as fragment_index for simplicity
                            0, # embedder_type (assuming 0 = default)
                         )
                        for chunk in chunks_to_write
                    ]
                    await conn.executemany(upsert_query, data_to_insert)
                    logger.debug(f"Upserted {len(chunks_to_write)} chunks into af_collab_embeddings for {record.object_id}")

                # Update indexed_at in af_collab table instead of using embedding_collab_index_state
                state_query = """
                UPDATE af_collab
                SET indexed_at = NOW()
                WHERE oid = $1;
                """
                await conn.execute(state_query, record.object_id)
                
                logger.info(f"Successfully wrote embeddings to af_collab_embeddings and updated state for {record.object_id}")

            except Exception as e:
                logger.exception(f"Database transaction failed for {record.object_id}: {e}")
                raise

# --- Main Processing Logic ---

async def process_document(doc_info: DocInfo, pool: asyncpg.Pool, embedder: Embedder, args: Args) -> Tuple[int, int]:
    """Processes a single document. Returns (success_count, failure_count)."""
    object_id = doc_info.object_id
    start_time = time.monotonic()
    logger.info(f"Processing document {object_id}")

    paragraphs = await get_document_paragraphs_placeholder(pool, object_id)
    if not paragraphs:
        logger.error(f"Failed to get content (using placeholder) for {object_id}. Skipping.")
        return (0, 1)

    existing_fragments = await get_collab_embedding_fragment_ids(pool, object_id)
    chunks = create_chunks(object_id, paragraphs, embedder.model_name())

    chunks_to_embed_count = 0
    for chunk in chunks:
        if chunk.fragment_id in existing_fragments:
            chunk.content = None
        else:
            chunks_to_embed_count += 1

    if chunks_to_embed_count == 0 and existing_fragments:
         logger.info(f"Document {object_id} already fully embedded based on existing fragments. Skipping embedding.")
         # Optionally, still update indexed_at state if needed, but skipping for simplicity
         return (1, 0)
    elif chunks_to_embed_count == 0 and not existing_fragments:
         logger.warning(f"Document {object_id} produced no chunks to embed and none existed. Skipping.")
         return (0, 1)

    logger.debug(f"Document {object_id}: Found {len(chunks)} total chunks, {chunks_to_embed_count} need embedding.")

    embedding_result = await embedder.embed(chunks)

    if embedding_result:
        logger.info(f"Embedding successful for {object_id}. Tokens used: {embedding_result.tokens_consumed}. Time: {time.monotonic() - start_time:.2f}s")
        record = EmbeddingRecord(
            workspace_id=doc_info.workspace_id,
            object_id=object_id,
            collab_type=str(args.collab_type_document),
            tokens_used=embedding_result.tokens_consumed,
            chunks=embedding_result.chunks,
        )
        try:
            await write_embeddings_to_db(pool, record)
            return (1, 0)
        except Exception:
            # Error logged within write_embeddings_to_db
            return (0, 1)
    else:
        logger.error(f"Embedding API call failed for {object_id}. Time: {time.monotonic() - start_time:.2f}s")
        return (0, 1)

async def fetch_unindexed_documents(pool: asyncpg.Pool, args: Args) -> List[DocInfo]:
    logger.info("Fetching list of unindexed documents...")
    base_query = f"""
        SELECT c.oid, c.workspace_id
        FROM af_collab c
        JOIN af_workspace w ON c.workspace_id = w.workspace_id
        WHERE c.partition_key = $1
          AND c.indexed_at IS NULL
          AND NOT COALESCE(w.settings->>'disable_search_indexing', 'false')::boolean
    """
    query_params = [args.collab_type_document]
    param_index = 2

    if args.workspace_id:
        base_query += f" AND c.workspace_id = ${param_index}"
        query_params.append(uuid.UUID(args.workspace_id))
        logger.info(f"Filtering by workspace_id: {args.workspace_id}")
        param_index += 1

    base_query += " ORDER BY c.created_at ASC" # Process older documents first

    if args.limit:
        base_query += f" LIMIT ${param_index}"
        query_params.append(args.limit)

    try:
        rows = await pool.fetch(base_query, *query_params)
        docs = [DocInfo(object_id=row['oid'], workspace_id=row['workspace_id']) for row in rows]
        logger.info(f"Found {len(docs)} unindexed documents matching criteria.")
        return docs
    except Exception as e:
        logger.exception(f"Failed to fetch unindexed documents: {e}")
        return []

async def main():
    args = parse_args()
    logger.info(f"Starting batch indexer with config: {args}")

    pool = None
    embedder = None
    total_processed = 0
    total_success = 0
    total_failed = 0

    try:
        logger.info("Initializing connections...")
        pool = await asyncpg.create_pool(args.database_url, min_size=1, max_size=args.max_concurrency + 1)
        if pool is None: raise ConnectionError("Failed to create DB pool")
        logger.info("Database pool initialized.")

        embedder = Embedder(args)
        await embedder.start()
        logger.info("Embedder initialized.")

        # --- Fetch documents --- 
        docs_to_process = await fetch_unindexed_documents(pool, args)
        if not docs_to_process:
             logger.info("No documents found to process. Exiting.")
             return

        # --- Process documents concurrently --- 
        semaphore = asyncio.Semaphore(args.max_concurrency)
        tasks = []

        async def process_with_semaphore(doc_info: DocInfo):
            async with semaphore:
                return await process_document(doc_info, pool, embedder, args)

        for doc_info in docs_to_process:
            tasks.append(asyncio.create_task(process_with_semaphore(doc_info)))

        logger.info(f"Dispatching {len(tasks)} documents for processing across {args.max_concurrency} concurrent workers...")
        results = await asyncio.gather(*tasks)

        for success, failed in results:
            total_success += success
            total_failed += failed
            total_processed += (success + failed)

        logger.info(f"Finished processing {total_processed} documents.")
        logger.info(f"Success: {total_success}, Failed: {total_failed}")

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    except Exception as e:
        logger.exception(f"Unhandled exception in main execution: {e}")
    finally:
        logger.info("Shutting down...")
        if embedder:
            await embedder.stop()
            logger.info("Embedder stopped.")
        if pool:
            await pool.close()
            logger.info("Database pool closed.")
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    # Set higher logging level for noisy libraries
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Script failed to run: {e}")
        exit(1) 