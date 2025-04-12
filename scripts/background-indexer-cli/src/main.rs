use appflowy_worker::indexer_worker::{run_background_indexer, BackgroundIndexerConfig};
use clap::Parser;
use database::pg_connection;
use indexer::metrics::EmbeddingMetrics;
use indexer::thread_pool::ThreadPoolNoAbort;
use indexer::vector::embedder::{AzureConfig, OpenAIConfig};
use redis::aio::ConnectionManager;
use sqlx::PgPool;
use std::sync::Arc;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long, env = "DATABASE_URL")]
    database_url: String,

    #[clap(long, env = "REDIS_URL")]
    redis_url: String,

    #[clap(long, env = "BACKGROUND_INDEXER_ENABLE", default_value = "true")]
    enable: bool,

    #[clap(long, env = "AZURE_AI_API_KEY")]
    azure_api_key: Option<String>,

    #[clap(long, env = "AZURE_AI_ENDPOINT")]
    azure_endpoint: Option<String>,

    #[clap(long, env = "AZURE_AI_DEPLOYMENT_ID")]
    azure_deployment_id: Option<String>,

    #[clap(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    #[clap(long, env = "OPENAI_API_BASE")]
    openai_api_base: Option<String>,

    #[clap(long, env = "OPENAI_MODEL")]
    openai_model: Option<String>,

    #[clap(long, env = "BACKGROUND_INDEXER_TICK_INTERVAL_SECS", default_value = "10")]
    tick_interval_secs: u64,

    #[clap(long, env = "BACKGROUND_INDEXER_THREADS", default_value = "4")]
    threads: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Initialize Database Pool
    let pg_pool = pg_connection::init_pg_pool(&args.database_url)
        .await
        .expect("Failed to initialize database pool");

    // Initialize Redis Client
    let redis_client = redis::Client::open(args.redis_url)?;
    let redis_conn_manager = ConnectionManager::new(redis_client)
        .await
        .expect("Failed to initialize redis connection manager");

    // Initialize Metrics
    let embed_metrics = Arc::new(EmbeddingMetrics::default());

    // Initialize Thread Pool
    let threads = Arc::new(ThreadPoolNoAbort::new(args.threads));

    // Configure Embedder
    let open_ai_config = if let (Some(key), Some(model)) = (args.openai_api_key, args.openai_model) {
        Some(OpenAIConfig {
            api_key: key,
            api_base: args.openai_api_base,
            model,
        })
    } else {
        None
    };

    let azure_ai_config = if let (Some(key), Some(endpoint), Some(deployment_id)) = (
        args.azure_api_key,
        args.azure_endpoint,
        args.azure_deployment_id,
    ) {
        Some(AzureConfig {
            api_key: key,
            endpoint,
            deployment_id,
        })
    } else {
        None
    };

    // Configure Background Indexer
    let config = BackgroundIndexerConfig {
        enable: args.enable,
        open_ai_config,
        azure_ai_config,
        tick_interval_secs: args.tick_interval_secs,
    };

    // Run the background indexer
    tracing::info!("Starting background indexer CLI...");
    run_background_indexer(
        pg_pool,
        redis_conn_manager,
        embed_metrics,
        threads,
        config,
    )
    .await;

    tracing::info!("Background indexer CLI finished.");
    Ok(())
} 