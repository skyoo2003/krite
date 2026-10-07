# Releasing

Every package has its own version and is released by its own tag, `<package>-vX.Y.Z`. Pushing the
tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml): it checks the tag
against the package's manifest and publishes that one package with Trusted Publishing.

| Package | Registry | Tag | Version in | Depends on |
|---|---|---|---|---|
| `krite-core` | crates.io | `krite-core-vX.Y.Z` | `crates/krite-core/Cargo.toml` | — |
| `krite-runtime` | crates.io | `krite-runtime-vX.Y.Z` | `crates/krite-runtime/Cargo.toml` | core |
| `krite-candle` | crates.io | `krite-candle-vX.Y.Z` | `crates/krite-candle/Cargo.toml` | runtime |
| `krite-server` | crates.io | `krite-server-vX.Y.Z` | `crates/krite-server/Cargo.toml` | core, runtime |
| `krite-cli` | crates.io | `krite-cli-vX.Y.Z` | `crates/krite-cli/Cargo.toml` | runtime, candle, server |
| `krite-bench` | PyPI | `krite-bench-vX.Y.Z` | `benchmarks/pyproject.toml` | — |
| `krite-train` | PyPI | `krite-train-vX.Y.Z` | `training/pyproject.toml` | krite-bench |
| `krite` | PyPI (wheels) | `krite-vX.Y.Z` | `python/pyproject.toml` | runtime, candle (built in) |

Crate-to-crate requirements live in the root `Cargo.toml` (`[workspace.dependencies]`, e.g.
`krite-core = { path = "crates/krite-core", version = "0.1.0" }`). `krite-train` requires
`krite-bench>=0.1,<0.2` in `training/pyproject.toml`. The model (`krite-0.15b-v1` on the Hugging
Face Hub) is not versioned with any package; it is uploaded by `scripts/publish-model.sh`. The `krite`
wheels and sdist embed the crates at the tagged commit (`python/` is the `krite-py` workspace member,
never published to crates.io), so no crate release has to come first.

## Version rules

All packages are 0.x, so under Cargo's and this project's semver:

- **Patch** (`0.1.0 → 0.1.1`): compatible. Dependents' requirements (`0.1.0` means `^0.1.0`) already
  accept it. Release only the package that changed.
- **Minor** (`0.1.x → 0.2.0`): breaking. Every crate that depends on it must take the new
  requirement, and because their public APIs expose its types (`krite-runtime` uses `krite-core`'s
  request and answer types, and so on down the table), each of them gets a minor bump too.
  `krite-cli` is a binary: any bump.

The build enforces the requirement side: bumping `krite-core` to 0.2.0 without changing its
requirement in `[workspace.dependencies]` fails every dependent with
`failed to select a version for the requirement krite-core = "^0.1.0"`.

## Release one package

1. Open a PR that changes only versions:
   - the package's `version` in its manifest;
   - for a crate: run `cargo check --workspace` and commit the updated `Cargo.lock` (the release
     job publishes with `--locked`);
   - for a Python package: run `uv lock` in its directory, and in `training/` too when `krite-bench`
     changed.
2. Merge it after CI passes.
3. Tag the merge commit on `main` and push the tag:
   ```bash
   git switch main && git pull --ff-only
   git tag krite-core-v0.1.1 && git push origin krite-core-v0.1.1
   gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
   ```

## Breaking release across crates

Bump the changed crate's minor, change its requirement in `[workspace.dependencies]`, and bump
every crate below it in dependency order, all in one PR. After the merge, push the tags **one at a
time, in this order, waiting for each run to finish**: `cargo publish` builds the package against
crates.io, so its dependencies must be published first.

```
krite-core → krite-runtime → krite-candle, krite-server → krite-cli
```

Example: a breaking change in `krite-core` 0.1.3.

| Crate | Before | After | Requirement in `[workspace.dependencies]` |
|---|---|---|---|
| `krite-core` | 0.1.3 | 0.2.0 | `version = "0.2.0"` |
| `krite-runtime` | 0.1.1 | 0.2.0 | `version = "0.2.0"` |
| `krite-candle` | 0.1.0 | 0.2.0 | `version = "0.2.0"` |
| `krite-server` | 0.1.2 | 0.2.0 | `version = "0.2.0"` |
| `krite-cli` | 0.1.4 | 0.2.0 | — |

Tags: `krite-core-v0.2.0`, then `krite-runtime-v0.2.0`, then `krite-candle-v0.2.0` and
`krite-server-v0.2.0`, then `krite-cli-v0.2.0`. The versions need not match; they match here
because each one was a minor bump.

For Python, release `krite-bench` first. A `krite-bench` minor (`0.2.0`) needs a `krite-train`
release that widens its range (`krite-bench>=0.2,<0.3`) only when `krite-train` uses the new API.

## Reruns and mistakes

- A failed run (network, registry outage): `gh run rerun <run-id> --failed`. Both jobs skip a version
  that is already published.
- A tag that does not match its manifest fails before anything is uploaded. Delete it
  (`git push origin :refs/tags/<tag>`, `git tag -d <tag>`), fix the version, tag again.
- A published version cannot be replaced. crates.io allows `cargo yank`; PyPI never accepts the same
  version again. Release the fix as the next patch.

## First release (once)

1. GitHub → Settings → Environments: create `pypi` and `crates-io`.
2. crates.io trusted publishing can only be configured on a crate that exists, so version 0.1.0 of
   the five crates is published by hand: `cargo login <token>` (scopes: publish-new,
   publish-update), `cargo publish --workspace` on `main`, then revoke the token.
3. crates.io → each crate → Settings → Trusted Publishing → GitHub: owner `skyoo2003`, repo `krite`,
   workflow `release.yml`, environment `crates-io`.
4. PyPI allows a pending publisher with a given owner, repo, workflow, and environment for one
   project at a time. Add it for `krite-bench` (environment `pypi`), push `krite-bench-v0.1.0`, then
   add it for `krite-train` and push `krite-train-v0.1.0`, then for `krite` and push `krite-v0.1.0`.
5. Model: `uvx --from huggingface_hub hf auth login`, then `scripts/publish-model.sh`.
