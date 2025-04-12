import argparse
import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import asyncpg
import redis.asyncio as redis
from dotenv import load_dotenv

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("batch_enqueue")


def get_env_var(name: str, default: str = None) -> str:
    return os.environ.get(name, default)

# --- Main Logic ---

async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enqueue existing documents for background indexing.")
    parser.add_argument("--database-url", default=get_env_var("DATABASE_URL"), required=get_env_var("DATABASE_URL") is None, help="PostgreSQL connection URL (e.g., postgresql://user:pass@host/db)")
    parser.add_argument("--redis-url", default=get_env_var("REDIS_URL"), required=get_env_var("REDIS_URL") is None, help="Redis connection URL")
    parser.add_argument("--redis-stream-key", default=get_env_var("BACKGROUND_INDEXER_STREAM_KEY", "background_embed_tasks"), help="Redis stream key for embedding tasks")
    parser.add_argument("--workspace-id", default=None, help="Optional: Process only a specific workspace UUID")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of documents to fetch from DB at a time")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Limit the total number of documents to enqueue")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks instead of enqueuing them")

    args = parser.parse_args()

    pool = None
    redis_client = None
    total_enqueued = 0
    total_fetched = 0
    collab_type_document = 0 # Based on code search

    try:
        logger.info(f"Connecting to Database: {args.database_url}")
        pool = await asyncpg.create_pool(args.database_url, min_size=1, max_size=2)
        if pool is None:
             raise ConnectionError("Failed to create database pool")
        logger.info("Database connection successful.")

        logger.info(f"Connecting to Redis: {args.redis_url}")
        redis_client = redis.from_url(args.redis_url)
        await redis_client.ping()
        logger.info("Redis connection successful.")

        base_query = f"""
            SELECT c.oid, c.workspace_id
            FROM af_collab c
            JOIN af_workspace w ON c.workspace_id = w.workspace_id
            WHERE c.partition_key = $1
              AND c.indexed_at IS NULL
              AND NOT COALESCE(w.settings->>'disable_search_indexing', 'false')::boolean
        """

        query_params = [collab_type_document]
        param_index = 2

        if args.workspace_id:
            base_query += f" AND c.workspace_id = ${param_index}"
            query_params.append(uuid.UUID(args.workspace_id))
            logger.info(f"Filtering by workspace_id: {args.workspace_id}")
            param_index += 1

        base_query += " ORDER BY c.created_at ASC" # Process older documents first

        async with pool.acquire() as conn:
            async with conn.transaction(): # Use transaction for cursor
                 cursor = conn.cursor(base_query, *query_params)
                 logger.info("Opened database cursor to fetch documents.")

                 while True:
                    if args.limit is not None and total_fetched >= args.limit:
                        logger.info(f"Reached specified limit of {args.limit} documents.")
                        break

                    fetch_count = args.batch_size
                    if args.limit is not None:
                         fetch_count = min(args.batch_size, args.limit - total_fetched)

                    rows = await cursor.fetch(fetch_count)
                    if not rows:
                        logger.info("No more documents found matching criteria.")
                        break

                    total_fetched += len(rows)
                    logger.info(f"Fetched {len(rows)} documents (Total fetched: {total_fetched})...")

                    tasks_to_enqueue = []
                    for row in rows:
                        object_id = str(row['oid'])
                        workspace_id = str(row['workspace_id'])
                        created_at_ts = time.time() # Use current time for enqueueing

                        # IMPORTANT: Enqueuing WITHOUT 'data' field.
                        # The worker (main.py) will need modification to fetch content.
                        task_payload = {
                            b"workspace_id": workspace_id.encode('utf-8'),
                            b"object_id": object_id.encode('utf-8'),
                            b"collab_type": str(collab_type_document).encode('utf-8'),
                            b"created_at": str(created_at_ts).encode('utf-8'),
                            # b"data": json.dumps({"paragraphs": ["Example content"]}) # No data field for now
                        }
                        tasks_to_enqueue.append(task_payload)

                    if args.dry_run:
                        logger.info("[Dry Run] Would enqueue tasks:")
                        for task in tasks_to_enqueue:
                            logger.info(f"  - { {k.decode(): v.decode() for k,v in task.items()} }")
                        total_enqueued += len(tasks_to_enqueue)
                    else:
                        # Enqueue in batches for efficiency
                        pipeline = redis_client.pipeline()
                        for task_payload in tasks_to_enqueue:
                             pipeline.xadd(args.redis_stream_key, task_payload)
                        results = await pipeline.execute()
                        success_count = sum(1 for r in results if r) # XADD returns message ID on success
                        total_enqueued += success_count
                        logger.info(f"Enqueued {success_count} tasks to Redis stream '{args.redis_stream_key}'")
                        if success_count < len(tasks_to_enqueue):
                            logger.warning(f"Failed to enqueue {len(tasks_to_enqueue) - success_count} tasks.")

                    if len(rows) < fetch_count:
                         logger.info("Fetched last batch of documents.")
                         break # Fetched less than requested, must be the end

    except (ConnectionRefusedError, asyncpg.exceptions.CannotConnectNowError) as e:
        logger.critical(f"Could not connect to Database or Redis: {e}")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        if pool:
            await pool.close()
            logger.info("Database pool closed.")
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed.")
        logger.info(f"Script finished. Total documents fetched: {total_fetched}. Total tasks enqueued (or simulated): {total_enqueued}.")

if __name__ == "__main__":
    asyncio.run(main()) 