<p align="center">
  <img src="https://reseam.app/logo.svg" alt="Reseam logo" width="96">
</p>

<h1 align="center">Reseam Translate</h1>

Translates posts for the [Reseam API](https://git.reseam.app/reseam/api). It runs Mozilla's [Firefox Translations](https://github.com/mozilla/translations) models, the ones Firefox uses for its built-in translation, on an engine written from scratch for CPUs. No GPU, no Marian, no Python at runtime. Only the API should talk to it.

- **Fast on a small server.** About 3 times the throughput of Bergamot on the same 4 vCPUs, with a p99 latency in milliseconds instead of seconds.
- **Small.** One memory-mapped copy of each model, shared by every thread. All 20 directions fit in under 1 GB, and idle models are dropped from memory.
- **Same quality.** The same int8 arithmetic as Marian's intgemm. chrF is within 0.1 of Bergamot.
- **Made for social posts.** URLs are kept, @mentions are restored, and spacing and line breaks survive.

## API

```text
GET  /health          status, instruction set, threads, loaded pairs, counters
GET  /v1/languages    {"pairs": [["es", "en"], ...], "pivot": "en"}
POST /v1/translate    {"from": "es", "to": "en", "texts": ["..."]} -> {"translations": ["..."]}
```

Pairs without a direct model pivot through English. With `TRANSLATE_TOKEN` set, every request except `/health` needs `Authorization: Bearer <token>`. Request texts are never logged.

## Run

```bash
docker build -t reseam-translate .
docker run -v translate-models:/models -e LANGUAGES=es,ja,zh -p 8080:8080 reseam-translate
```

The image installs the latest build from this repository's releases. On start it downloads the models for `LANGUAGES` from Mozilla into the `/models` volume, checks their SHA-256, and keeps them between restarts. It defaults to 18 languages with `PRELOAD=1`.

Every push to `main` builds and tests on the CI runner, replaces the rolling `latest` release with `reseam-translate-linux-x64.tar.gz`, and redeploys through the `DOKPLOY_WEBHOOK` secret.

## Configure

Each setting is an environment variable, or the flag in parentheses.

| Var | Default | |
| --- | --- | --- |
| `MODELS_DIR` (`--models`) | `./models` | folder of `<from>-<to>/` model folders |
| `HOST`, `PORT` (`--host`, `--port`) | `0.0.0.0`, `8080` | |
| `THREADS` (`--threads`) | all cores | compute threads, one per physical core |
| `MAX_BATCH` (`--max-batch`) | `64` | sentences decoded together per model |
| `MAX_TEXTS` (`--max-texts`) | `50` | texts per request |
| `MAX_CHARS` (`--max-chars`) | `20000` | bytes of text per request |
| `HTTP_THREADS` (`--http-threads`) | `64` | requests held open at once |
| `PIN_FROM_CORE` (`--pin-from-core`) | off | pins compute thread `i` to core `N + i` |
| `PRELOAD` (`--preload`) | off | loads every model at startup |
| `IDLE_SECONDS` (`--idle-seconds`) | `300` | drops an unused model's pages after this long, `0` keeps all resident |
| `TRANSLATE_TOKEN` | empty | bearer token required on requests |

## Benchmarks

One 4-vCPU cloud VM (Xeon Sapphire Rapids, 2.1 GHz, 15 GB RAM) with all 20 directions loaded, traffic spread across them, and 64 concurrent clients. Latency is one post at a time on an idle server. Medians of three runs.

**WMT24++ social posts.** 5,264 posts over 16 directions, median 66 characters, chrF against human references.

| Engine | Posts/s | p50 / p99 | Memory | chrF |
| --- | --- | --- | --- | --- |
| Bergamot (Marian + intgemm), 4 workers | 199 | 22 / 1,958 ms | 13.1 GB | 46.31 |
| Bergamot, 2 workers | 97 | 19 / 1,203 ms | 7.3 GB | |
| Bergamot, 1 worker | 48 | 19 / 70 ms | 4.0 GB | |
| **Reseam Translate, AMX** | **619** | **2.7 / 9.7 ms** | **0.89 GB** | **46.22** |
| **Reseam Translate, AVX-512 VNNI** | **544** | **3.0 / 11.8 ms** | **0.89 GB** | **46.22** |
| **Reseam Translate, AVX2** | **234** | **5.5 / 21 ms** | **0.89 GB** | |

**Real posts from X and Reddit.** 2,699 posts over 18 directions, median 154 characters. 23% have links, 59% mentions, 39% line breaks.

| Engine | Posts/s | p50 / p99 | Memory |
| --- | --- | --- | --- |
| Bergamot, 4 workers | out of memory | | over 15 GB |
| Bergamot, 3 workers | 54 | 48 / 1,562 ms | 11.9 GB |
| **Reseam Translate, AMX** | **216** | **6.4 / 52 ms** | **0.99 GB** |

Firefox's WASM build of Bergamot cannot serve this: its 2 GB heap holds three models per worker. With only three models it does 24 posts/s, against 1,013 here.

Bergamot keeps a copy of every model per worker, about 3.3 GB each here. Of this engine's 0.89 GB, 0.19 GB is heap and 0.7 GB mapped weights. With every model idle the process holds about 140 MB, and an idle model pages back in within about 30 ms.

Output is not byte-identical to Bergamot, since floats are rounded in a different order and the odd near-tie flips. The chrF above is without mention restoring: WMT24++ replaced handles with `@user` and translated them in its references, so keeping handles costs 0.25 chrF there.

4-bit weights do not pay off. Re-quantizing to 4 bits in groups of 32 costs 0.9 chrF to save about 10 MB per model, and 4-bit embeddings drop chrF from 46 to 28.

## How it works

The models are Marian transformers: a 6-layer encoder, a 2- or 4-layer SSRU decoder, width 256 or 384, int8 weights with fixed activation scales.

- **No framework.** A fixed forward pass over preallocated buffers, with no graph and no allocation per batch. Under load Marian spends 63% of its time outside its int8 kernels.
- **Kernels per instruction set,** picked at runtime: AVX2, AVX-VNNI, AVX-512 VNNI and AMX. Weights are packed once into a layout all of them read natively. One binary runs from Haswell and Zen 1 to Sapphire Rapids and Zen 5.
- **Fused epilogues.** Dequantization, bias, ReLU and requantization happen inside the matrix product, so the FFN's hidden layer never exists in float. Q, K and V, the SSRU gates, and cross-attention's K and V are each one product.
- **Continuous batching.** Sentences join and leave the batch at every decoder step, across all models, so a request never waits for a batch to finish.
- **Two ways of splitting work.** With little work, every product is split by columns so one sentence uses every core. With enough work, each thread runs a contiguous share of all rows of all models without a barrier.
- **Per-sentence output layer.** Each sentence scores only its own shortlist of about 1,200 words. A post translates the same alone or among a thousand.
- **Attention on transposed keys,** so one vector instruction scores 16 keys.
- **Memory-mapped weights.** The first load packs a model and writes `model.reseam` next to it. Later loads map that file, and 16 models load in under a second. Processes share the pages and the kernel can reclaim them.
- **One working memory.** Activations live in one block shared by every model, released when the engine goes idle.

Around the model, URLs are cut out and the text around them translated on its own. When a translation has as many mentions as its source, each goes back to the original handle. Long sentences are translated in pieces.

There is no beam search (Firefox decodes greedily too), no alignments, no HTML and no quality scores.

## Develop

Needs CMake 3.28+ and GCC 13+ or Clang 17+. SentencePiece, cpp-httplib and nlohmann/json are fetched at configure time.

```bash
cmake -S . -B build -G Ninja && cmake --build build
ctest --test-dir build                          # kernels, attention and text, on every ISA the CPU has
python3 scripts/fetch-models.py models es,ja,zh
MODELS_DIR=models build/reseam-translate
curl -d '{"from":"es","to":"en","texts":["Hola"]}' localhost:8080/v1/translate
```

The build also makes:

| Tool | Does |
| --- | --- |
| `bench <models> <corpus.tsv> <n> <threads> <out.tsv> [clients] [maxBatch]` | latency and throughput on `from<TAB>to<TAB>text` lines. `BENCH_SKIP_LATENCY=1` skips latency, `BENCH_PRELOAD=1` loads every model first |
| `bench_gemm` | the int8 kernels at the models' shapes, in multiply-adds per cycle |
| `translate <models> <from> <to>` | translates stdin line by line |

`RESEAM_MT_ISA=avx2|avxvnni|avx512vnni|amx|reference` forces an instruction set. `RESEAM_MT_PROFILE=1` prints compute time per kind of operation at exit. `RESEAM_MT_DEBUG=1` prints per-layer values in the format of Marian's `debug()`.

Built with C++20, SentencePiece and cpp-httplib.

## Models

The models are © Mozilla, under the [MPL 2.0](https://mozilla.org/MPL/2.0/).
