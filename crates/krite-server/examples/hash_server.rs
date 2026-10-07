//! Serves the weight-free `HashBackend` (model id `hash`) on 127.0.0.1, for the Jev SDK conformance
//! suite (`scripts/check-compat.sh`). Usage: `cargo run -p krite-server --example hash_server -- [port]`.

#[path = "../tests/support/hash.rs"]
mod hash;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port = std::env::args().nth(1).map_or(Ok(8199), |p| p.parse())?;
    krite_server::serve(krite_runtime::Runtime::new(hash::HashBackend, 1 << 20), port).await
}
