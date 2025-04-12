import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import asyncpg
import redis.asyncio as redis
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration & Argument Parsing ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("background_indexer")


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
    redis_url: str
    enable: bool = True
    openai_config: Optional[OpenAIConfig] = None
    azure_config: Optional[AzureConfig] = None
    tick_interval_secs: int = 10
    batch_size: int = 10 # Number of tasks to read from Redis at once
    max_concurrency: int = 4 # Max parallel embedding tasks
    redis_stream_key: str = "background_embed_tasks"
    redis_group_name: str = "indexer_group"
    redis_consumer_name: str = f"consumer-{uuid.uuid4()}"


def parse_args() -> Args:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AppFlowy Background Indexer (Python)")

    parser.add_argument("--database-url", default=get_env_var("DATABASE_URL"), required=get_env_var("DATABASE_URL") is None, help="PostgreSQL connection URL")
    parser.add_argument("--redis-url", default=get_env_var("REDIS_URL"), required=get_env_var("REDIS_URL") is None, help="Redis connection URL")
    parser.add_argument("--disable", action="store_false", dest="enable", help="Disable the indexer")
    parser.add_argument("--tick-interval-secs", type=int, default=get_env_var("BACKGROUND_INDEXER_TICK_INTERVAL_SECS", 10), help="Interval between checking for new tasks")
    parser.add_argument("--batch-size", type=int, default=get_env_var("BACKGROUND_INDEXER_BATCH_SIZE", 10), help="Number of tasks to fetch from Redis at once")
    parser.add_argument("--max-concurrency", type=int, default=get_env_var("BACKGROUND_INDEXER_THREADS", 4), help="Maximum concurrent embedding tasks")
    parser.add_argument("--redis-stream-key", default=get_env_var("BACKGROUND_INDEXER_STREAM_KEY", "background_embed_tasks"))
    parser.add_argument("--redis-group-name", default=get_env_var("BACKGROUND_INDEXER_GROUP_NAME", "indexer_group"))

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

    if not openai_cfg and not azure_cfg and parsed.enable:
        parser.error("Either OpenAI or Azure AI configuration must be provided if enabled.")

    return Args(
        database_url=parsed.database_url,
        redis_url=parsed.redis_url,
        enable=parsed.enable,
        openai_config=openai_cfg,
        azure_config=azure_cfg,
        tick_interval_secs=parsed.tick_interval_secs,
        batch_size=parsed.batch_size,
        max_concurrency=parsed.max_concurrency,
        redis_stream_key=parsed.redis_stream_key,
        redis_group_name=parsed.redis_group_name,
    )

# --- Data Structures ---

class UnindexedData(BaseModel):
    paragraphs: Optional[List[str]] = None
    text: Optional[str] = None

    def get_paragraphs(self) -> List[str]:
        if self.paragraphs:
            return self.paragraphs
        if self.text:
            # Simple split, might need refinement to match Rust logic
            return self.text.split('\n')
        return [] # Should not happen based on Rust code

class UnindexedCollabTask(BaseModel):
    stream_id: str
    workspace_id: str
    object_id: str
    collab_type: str
    created_at: datetime
    data: Optional[UnindexedData]

    @classmethod
    def from_redis_message(cls, stream_id: str, message: Dict[bytes, bytes]) -> Optional['UnindexedCollabTask']:
        try:
            decoded_message = {k.decode('utf-8'): v.decode('utf-8') for k, v in message.items()}
            workspace_id = decoded_message.get('workspace_id')
            object_id = decoded_message.get('object_id')
            collab_type = decoded_message.get('collab_type')
            created_at_ts_str = decoded_message.get('created_at')
            data_json = decoded_message.get('data')

            if not all([workspace_id, object_id, collab_type, created_at_ts_str]):
                logger.warning(f"Missing required base fields in message {stream_id}: {decoded_message}")
                return None

            created_at = datetime.fromtimestamp(float(created_at_ts_str), tz=timezone.utc)
            data_obj = None
            if data_json:
                 try:
                    data_obj = UnindexedData.parse_raw(data_json)
                 except Exception as parse_error:
                     logger.error(f"Failed to parse 'data' field for message {stream_id}: {parse_error}. Data: {data_json}")
                     # Decide if you want to proceed without data or fail
                     # Proceeding without data for now, it will be fetched later if needed.

            return cls(
                stream_id=stream_id,
                workspace_id=workspace_id,
                object_id=object_id,
                collab_type=collab_type,
                created_at=created_at,
                data=data_obj,
            )
        except Exception as e:
            logger.error(f"Failed to parse Redis message structure {stream_id}: {message}. Error: {e}")
            return None

@dataclass
class Chunk:
    fragment_id: str # Typically f"{paragraph_index}:{chunk_index}"
    object_id: str
    content: Optional[str]
    embedding: Optional[List[float]] = None
    chunk_index: int = 0 # Internal tracking, not stored in DB
    paragraph_index: int = 0 # Internal tracking, not stored in DB

@dataclass
class EmbeddingResult:
    tokens_consumed: int
    chunks: List[Chunk]

@dataclass
class EmbeddingRecord:
    workspace_id: str
    object_id: str
    collab_type: str
    tokens_used: int
    chunks: List[Chunk]


# --- Embedder Logic ---

class Embedder:
    def __init__(self, args: Args):
        self.args = args
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()

    def model_name(self) -> str:
        if self.args.openai_config:
            return self.args.openai_config.model
        if self.args.azure_config:
            # Azure doesn't expose model name easily via config, return deployment id
            return self.args.azure_config.deployment_id
        return "unknown"

    async def embed(self, chunks_to_embed: List[Chunk]) -> Optional[EmbeddingResult]:
        if not chunks_to_embed:
            return EmbeddingResult(tokens_consumed=0, chunks=[])

        texts = [chunk.content for chunk in chunks_to_embed if chunk.content is not None]
        if not texts:
            # All chunks were marked as unchanged
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
            async with self.session.post(url, json=payload, headers=headers) as response:
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
                            # Decide how to handle: skip chunk, fail task? Skipping for now.
                    final_chunks.append(chunk)

                return EmbeddingResult(tokens_consumed=tokens, chunks=final_chunks)

        except aiohttp.ClientError as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing OpenAI response: {e}")
            return None

    async def _embed_azure(self, texts: List[str], original_chunks: List[Chunk]) -> Optional[EmbeddingResult]:
        config = self.args.azure_config
        if not config or not self.session:
            return None

        # Construct Azure URL (example, adjust if needed)
        url = f"{config.endpoint}/openai/deployments/{config.deployment_id}/embeddings?api-version=2023-05-15"
        headers = {"api-key": config.api_key}
        payload = {"input": texts}

        try:
            async with self.session.post(url, json=payload, headers=headers) as response:
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
                             # Decide how to handle: skip chunk, fail task? Skipping for now.
                    final_chunks.append(chunk)

                return EmbeddingResult(tokens_consumed=tokens, chunks=final_chunks)
        except aiohttp.ClientError as e:
            logger.error(f"Azure API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing Azure response: {e}")
            return None

# --- Chunking Logic (Simplified) ---
# TODO: Refine this to match Rust's indexer logic if possible

def create_chunks(object_id: str, paragraphs: List[str], model: str) -> List[Chunk]:
    chunks = []
    # Very basic chunking: treat each paragraph as a chunk for simplicity
    # A more sophisticated approach would use token counting (e.g., tiktoken)
    # and split paragraphs if they exceed a token limit, matching Rust's behavior.
    for i, para in enumerate(paragraphs):
        if para.strip(): # Avoid empty paragraphs
            fragment_id = f"{i}:0" # Simple fragment ID: paragraph_index:chunk_index_within_para
            chunks.append(Chunk(
                fragment_id=fragment_id,
                object_id=object_id,
                content=para, # Initially set content for embedding
                paragraph_index=i,
                chunk_index=0
            ))
    return chunks

# --- Database Operations ---

async def get_collabs_indexed_at(pool: asyncpg.Pool, collab_ids: List[str]) -> Dict[str, datetime]:
    if not collab_ids:
        return {}
    query = """
    SELECT object_id, indexed_at
    FROM embedding_collab_index_state
    WHERE object_id = ANY($1::varchar[])
    """
    try:
        rows = await pool.fetch(query, collab_ids)
        return {row['object_id']: row['indexed_at'] for row in rows}
    except Exception as e:
        logger.error(f"Failed to get collab indexed_at: {e}")
        return {}

async def get_collab_embedding_fragment_ids(pool: asyncpg.Pool, collab_ids: List[str]) -> Dict[str, set[str]]:
    if not collab_ids:
        return {}
    # Assuming a table structure like embedding_chunks(object_id, fragment_id, ...)
    query = """
    SELECT object_id, fragment_id
    FROM embedding_chunks -- Adjust table name if different
    WHERE object_id = ANY($1::varchar[])
    """
    results = {}
    try:
        rows = await pool.fetch(query, collab_ids)
        for row in rows:
            obj_id = row['object_id']
            frag_id = row['fragment_id']
            if obj_id not in results:
                results[obj_id] = set()
            results[obj_id].add(frag_id)
        return results
    except Exception as e:
        logger.error(f"Failed to get collab embedding fragment ids: {e}")
        return {}

async def write_embeddings_to_db(pool: asyncpg.Pool, record: EmbeddingRecord):
    # This needs to be an atomic operation (transaction)
    # 1. Delete existing chunks for the object_id that are NOT in the new record's chunk list
    # 2. Upsert the new/updated chunks
    # 3. Update the embedding_collab_index_state table
    async with pool.acquire() as conn:
        async with conn.transaction():
            try:
                # Get existing fragment IDs for this object
                existing_query = "SELECT fragment_id FROM embedding_chunks WHERE object_id = $1"
                existing_rows = await conn.fetch(existing_query, record.object_id)
                existing_fragment_ids = {row['fragment_id'] for row in existing_rows}

                new_fragment_ids = {chunk.fragment_id for chunk in record.chunks}

                # Chunks to delete
                to_delete = existing_fragment_ids - new_fragment_ids
                if to_delete:
                    delete_query = "DELETE FROM embedding_chunks WHERE object_id = $1 AND fragment_id = ANY($2::varchar[])"
                    await conn.execute(delete_query, record.object_id, list(to_delete))
                    logger.debug(f"Deleted {len(to_delete)} old chunks for {record.object_id}")

                # Chunks to upsert
                chunks_to_write = [chunk for chunk in record.chunks if chunk.embedding is not None]
                if chunks_to_write:
                    upsert_query = """
                    INSERT INTO embedding_chunks (object_id, workspace_id, collab_type, fragment_id, chunk, embedding, token_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (object_id, fragment_id)
                    DO UPDATE SET
                        chunk = EXCLUDED.chunk,
                        embedding = EXCLUDED.embedding,
                        token_count = EXCLUDED.token_count,
                        updated_at = NOW();
                    """
                    # Estimate token count simply by content length for now
                    # TODO: Use tiktoken for accurate count if needed
                    data_to_insert = [
                        (
                            chunk.object_id,
                            record.workspace_id,
                            record.collab_type,
                            chunk.fragment_id,
                            chunk.content,
                            chunk.embedding,
                            len(chunk.content) if chunk.content else 0
                         )
                        for chunk in chunks_to_write
                    ]
                    await conn.executemany(upsert_query, data_to_insert)
                    logger.debug(f"Upserted {len(chunks_to_write)} chunks for {record.object_id}")

                # Update index state
                state_query = """
                INSERT INTO embedding_collab_index_state (object_id, workspace_id, indexed_at, tokens_indexed)
                VALUES ($1, $2, NOW(), $3)
                ON CONFLICT (object_id)
                DO UPDATE SET
                    indexed_at = NOW(),
                    tokens_indexed = embedding_collab_index_state.tokens_indexed + EXCLUDED.tokens_indexed;
                 """
                await conn.execute(state_query, record.object_id, record.workspace_id, record.tokens_used)
                logger.info(f"Successfully wrote embeddings for {record.object_id}, tokens: {record.tokens_used}")

            except Exception as e:
                logger.error(f"Database transaction failed for {record.object_id}: {e}")
                # Transaction automatically rolls back
                raise # Re-raise to signal failure

async def fetch_document_content(pool: asyncpg.Pool, object_id: str) -> Optional[List[str]]:
    """Placeholder function to fetch document content.

    Attempts to fetch the raw data blob from af_collab.
    Actual implementation needs to handle decoding this blob into paragraphs,
    which is complex and likely requires replicating Rust logic.
    Returns None if content cannot be retrieved or decoded by this placeholder.
    """
    logger.info(f"Task {object_id} missing data, attempting to fetch content from DB.")
    try:
        # Assuming partition_key 0 is Document
        # Assuming 'data' column holds the collab blob
        query = "SELECT data FROM af_collab WHERE oid = $1 AND partition_key = 0"
        row = await pool.fetchrow(query, uuid.UUID(object_id))

        if row and row['data']:
            # We found the blob, but decoding it here is complex.
            # In a real scenario, you would call Rust code or a dedicated service
            # to decode this blob (row['data']) into paragraphs.
            logger.warning(f"Found data blob for {object_id}, but decoding logic is NOT IMPLEMENTED in this script. Skipping task.")
            # For demonstration, return None. Replace this with actual decoding if possible.
            return None
        elif row:
            logger.error(f"Found collab row for {object_id}, but 'data' column is null or missing.")
            return None
        else:
            logger.error(f"Could not find collab row for object_id {object_id} in af_collab table.")
            return None
    except Exception as e:
        logger.error(f"Database error fetching content for {object_id}: {e}")
        return None


# --- Redis Operations ---

async def ensure_redis_group(redis_client: redis.Redis, stream_key: str, group_name: str):
    try:
        await redis_client.xgroup_create(stream_key, group_name, id='0', mkstream=True)
        logger.info(f"Consumer group '{group_name}' created for stream '{stream_key}'.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP Consumer Group name already exists" in str(e):
            logger.info(f"Consumer group '{group_name}' already exists for stream '{stream_key}'.")
        else:
            logger.error(f"Failed to create/ensure consumer group '{group_name}': {e}")
            raise

async def read_tasks_from_redis(redis_client: redis.Redis, args: Args) -> AsyncGenerator[List[UnindexedCollabTask], None]:
    stream_dict = {args.redis_stream_key: '>'}
    while True:
        try:
            # Read up to batch_size messages, block for a short time if none
            response = await redis_client.xreadgroup(
                args.redis_group_name,
                args.redis_consumer_name,
                stream_dict,
                count=args.batch_size,
                block=args.tick_interval_secs * 1000 # Block in milliseconds
            )

            if not response:
                logger.debug("No new messages in Redis stream for this consumer.")
                yield [] # No new messages
                continue

            tasks = []
            message_ids_to_ack = []
            # Response format: [[b'stream_key', [(b'message_id', {b'field': b'value'})]]]
            for stream_result in response:
                stream_key, messages = stream_result
                if stream_key.decode('utf-8') == args.redis_stream_key:
                    for message_id_bytes, message_data in messages:
                        stream_id = message_id_bytes.decode('utf-8')
                        task = UnindexedCollabTask.from_redis_message(stream_id, message_data)
                        if task:
                            tasks.append(task)
                            message_ids_to_ack.append(stream_id)
                        else:
                            # Message failed parsing, ACK it anyway to remove it
                            message_ids_to_ack.append(stream_id)
                            logger.warning(f"Skipping unparseable message {stream_id}")

            if tasks:
                 logger.info(f"Read {len(tasks)} tasks from Redis stream")
                 yield tasks
            else:
                # This case might happen if all messages read were unparseable
                logger.debug("Read messages, but none were parseable tasks.")
                yield []

            # ACK processed/skipped messages
            if message_ids_to_ack:
                try:
                    await redis_client.xack(args.redis_stream_key, args.redis_group_name, *message_ids_to_ack)
                    logger.debug(f"ACKed {len(message_ids_to_ack)} messages.")
                except Exception as e: # Catch broad exception for ACK failure
                    logger.error(f"Failed to ACK messages: {message_ids_to_ack}. Error: {e}")
                    # Decide how to handle ACK failure (retry? log and continue?)

        except redis.exceptions.ConnectionError as e: # Corrected path
            logger.error(f"Redis connection error: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except redis.exceptions.TimeoutError:
             logger.debug("Redis read timed out (expected during blocking wait). Will retry.")
             yield [] # No messages received
        except redis.exceptions.RedisError as e: # Catch other specific Redis errors
            logger.error(f"Redis error reading stream: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e: # Catch any other unexpected errors
            logger.exception(f"Unexpected error reading from Redis stream: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

# --- Main Processing Logic ---

async def process_task(task: UnindexedCollabTask, pool: asyncpg.Pool, embedder: Embedder, existing_fragments: Optional[set[str]]) -> Optional[EmbeddingRecord]:
    start_time = time.monotonic()
    logger.info(f"Processing task for object {task.object_id} ({task.collab_type})")

    paragraphs = None
    if task.data:
        paragraphs = task.data.get_paragraphs()
    else:
        # Data was missing, attempt to fetch it
        fetched_paragraphs = await fetch_document_content(pool, task.object_id)
        if fetched_paragraphs:
            paragraphs = fetched_paragraphs
        else:
            logger.warning(f"Could not get/process content for task {task.object_id} (data was missing and fetch failed/not implemented). Skipping.")
            return None # Skip task if content cannot be obtained

    if not paragraphs:
        logger.warning(f"Task {task.object_id} has no paragraphs/text to process (either initially or after fetch attempt).")
        return None

    chunks = create_chunks(task.object_id, paragraphs, embedder.model_name())

    if existing_fragments:
        chunks_to_embed_count = 0
        for chunk in chunks:
            if chunk.fragment_id in existing_fragments:
                chunk.content = None # Mark as unchanged, don't embed
            else:
                chunks_to_embed_count += 1
        logger.debug(f"Object {task.object_id}: Found {len(chunks)} total chunks, {chunks_to_embed_count} need embedding.")
    else:
        logger.debug(f"Object {task.object_id}: No existing fragments found, embedding all {len(chunks)} chunks.")

    embedding_result = await embedder.embed(chunks)

    if embedding_result:
        logger.info(f"Embedding successful for {task.object_id}. Tokens used: {embedding_result.tokens_consumed}. Time: {time.monotonic() - start_time:.2f}s")
        return EmbeddingRecord(
            workspace_id=task.workspace_id,
            object_id=task.object_id,
            collab_type=task.collab_type,
            tokens_used=embedding_result.tokens_consumed,
            chunks=embedding_result.chunks,
        )
    else:
        logger.error(f"Embedding failed for {task.object_id}. Time: {time.monotonic() - start_time:.2f}s")
        # Increment failure metric here if implementing metrics
        return None

async def worker(id: int, task_queue: asyncio.Queue, result_queue: asyncio.Queue, pool: asyncpg.Pool, embedder: Embedder):
    logger.info(f"Worker {id} started.")
    while True:
        task_batch = await task_queue.get()
        if task_batch is None: # Sentinel value to stop worker
            logger.info(f"Worker {id} stopping.")
            break

        tasks_to_process: List[UnindexedCollabTask]
        existing_fragments_map: Dict[str, set[str]]
        tasks_to_process, existing_fragments_map = task_batch

        processed_results = []
        for task in tasks_to_process:
            existing_frags = existing_fragments_map.get(task.object_id)
            try:
                record = await process_task(task, pool, embedder, existing_frags)
                if record:
                    processed_results.append(record)
            except Exception as e:
                logger.error(f"Worker {id}: Unhandled exception processing task {task.object_id}: {e}")
                # Consider adding failure metric/tracking

        if processed_results:
            await result_queue.put(processed_results)

        task_queue.task_done()

async def db_writer(result_queue: asyncio.Queue, pool: asyncpg.Pool):
    logger.info("DB Writer started.")
    while True:
        records_batch = await result_queue.get()
        if records_batch is None: # Sentinel value
            logger.info("DB Writer stopping.")
            break

        logger.info(f"DB Writer received {len(records_batch)} records to write.")
        # Write records one by one with transactions for robustness
        success_count = 0
        fail_count = 0
        for record in records_batch:
            try:
                await write_embeddings_to_db(pool, record)
                success_count += 1
            except Exception:
                # Error already logged in write_embeddings_to_db
                fail_count += 1
                # Add failure metric here if needed

        logger.info(f"DB Writer finished batch. Success: {success_count}, Failed: {fail_count}")
        result_queue.task_done()

async def main():
    args = parse_args()
    if not args.enable:
        logger.info("Background indexer is disabled via args.")
        return

    logger.info(f"Starting background indexer with config: {args}")

    pool = None
    redis_client = None
    embedder = None
    workers = []
    db_writer_task = None

    try:
        # --- Initialization ---
        logger.info("Initializing connections...")
        pool = await asyncpg.create_pool(args.database_url, min_size=1, max_size=args.max_concurrency + 2) # +1 for main, +1 for writer
        if pool is None:
            raise ConnectionError("Failed to create database pool")
        logger.info("Database pool initialized.")

        redis_client = redis.from_url(args.redis_url)
        await redis_client.ping() # Test connection
        logger.info("Redis connection successful.")

        await ensure_redis_group(redis_client, args.redis_stream_key, args.redis_group_name)

        embedder = Embedder(args)
        await embedder.start()
        logger.info("Embedder initialized.")

        # --- Setup Queues and Workers ---
        task_queue = asyncio.Queue(maxsize=args.max_concurrency * 2) # Buffer tasks
        result_queue = asyncio.Queue(maxsize=args.max_concurrency * 2) # Buffer results

        # Start DB Writer
        db_writer_task = asyncio.create_task(db_writer(result_queue, pool))

        # Start Workers
        for i in range(args.max_concurrency):
            task = asyncio.create_task(worker(i, task_queue, result_queue, pool, embedder))
            workers.append(task)

        # --- Main Loop: Read from Redis, Filter, Dispatch --- 
        logger.info("Starting main processing loop...")
        async for tasks in read_tasks_from_redis(redis_client, args):
            if not tasks:
                continue

            collab_ids = [task.object_id for task in tasks]

            # Filter tasks based on indexed_at timestamp
            indexed_at_map = await get_collabs_indexed_at(pool, collab_ids)
            filtered_tasks = []
            original_count = len(tasks)
            for task in tasks:
                indexed_at = indexed_at_map.get(task.object_id)
                if indexed_at is None or task.created_at > indexed_at:
                    filtered_tasks.append(task)
                else:
                    logger.debug(f"Skipping task {task.object_id} (created_at {task.created_at} <= indexed_at {indexed_at})")

            if not filtered_tasks:
                logger.info(f"Filtered out all {original_count} tasks based on indexed_at timestamp.")
                continue

            filtered_count = len(filtered_tasks)
            if filtered_count < original_count:
                logger.info(f"Filtered out {original_count - filtered_count} tasks based on indexed_at timestamp.")

            # Get existing fragments for remaining tasks
            collab_ids_to_fetch = [task.object_id for task in filtered_tasks]
            existing_fragments_map = await get_collab_embedding_fragment_ids(pool, collab_ids_to_fetch)

            # Dispatch tasks to workers
            # Could potentially batch tasks further here if needed
            await task_queue.put((filtered_tasks, existing_fragments_map))
            logger.info(f"Dispatched {len(filtered_tasks)} tasks to worker queue.")

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    except Exception as e:
        logger.exception(f"Unhandled exception in main loop: {e}")
    finally:
        logger.info("Shutting down...")
        # --- Cleanup ---
        if task_queue:
            # Signal workers to stop
            for _ in range(args.max_concurrency):
                await task_queue.put(None)
            # Wait for workers to finish processing inflight tasks
            await task_queue.join()
            logger.info("Worker queue joined.")

        if result_queue:
            # Signal DB writer to stop
            await result_queue.put(None)
            # Wait for writer to finish inflight writes
            await result_queue.join()
            logger.info("Result queue joined.")

        # Wait for worker tasks to complete
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
            logger.info("Worker tasks finished.")
        # Wait for DB writer task to complete
        if db_writer_task:
            await db_writer_task # Should already be finished after queue join
            logger.info("DB writer task finished.")

        if embedder:
            await embedder.stop()
            logger.info("Embedder stopped.")
        if pool:
            await pool.close()
            logger.info("Database pool closed.")
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed.")

        logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Script failed to run: {e}")
        exit(1) 