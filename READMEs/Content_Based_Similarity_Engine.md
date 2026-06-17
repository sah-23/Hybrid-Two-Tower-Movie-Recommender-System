# Content-Based Movie Similarity Engine

## Technical Report — Semantic Embedding Search with SBERT

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [System Architecture at a Glance](#3-system-architecture-at-a-glance)
4. [Data Preparation Pipeline](#4-data-preparation-pipeline)
   - 4.1 [Raw Data Loading and Merging](#41-raw-data-loading-and-merging)
   - 4.2 [JSON Field Parsing](#42-json-field-parsing)
   - 4.3 [Rich Text Description Construction](#43-rich-text-description-construction)
   - 4.4 [SBERT Encoding and L2 Normalization](#44-sbert-encoding-and-l2-normalization)
   - 4.5 [Metadata Table Construction](#45-metadata-table-construction)
   - 4.6 [Sanity Checks](#46-sanity-checks)
   - 4.7 [Saved Artifacts](#47-saved-artifacts)
5. [The Similarity Engine: Design and Mathematics](#5-the-similarity-engine-design-and-mathematics)
   - 5.1 [Core Principle: Dot Product on Normalized Vectors](#51-core-principle-dot-product-on-normalized-vectors)
   - 5.2 [Query Movie Lookup (Three-Tier Matching)](#52-query-movie-lookup-three-tier-matching)
   - 5.3 [Similarity Scoring Against the Full Catalog](#53-similarity-scoring-against-the-full-catalog)
   - 5.4 [Year Range Filtering with Score Masking](#54-year-range-filtering-with-score-masking)
   - 5.5 [Top-N Retrieval and Result Formatting](#55-top-n-retrieval-and-result-formatting)
6. [Production Inference Module](#6-production-inference-module)
   - 6.1 [Lazy Singleton Loading](#61-lazy-singleton-loading)
   - 6.2 [The Inference Bundle](#62-the-inference-bundle)
   - 6.3 [find_similar() — Production Version](#63-find_similar----production-version)
   - 6.4 [search_movies() — Title Search](#64-search_movies----title-search)
7. [Supporting Utilities](#7-supporting-utilities)
   - 7.1 [Popular Movies](#71-popular-movies)
   - 7.2 [TMDB Search (Notebook Version)](#72-tmdb-search-notebook-version)
8. [Empirical Similarity Results](#8-empirical-similarity-results)
9. [Key Design Decisions and Trade-offs](#9-key-design-decisions-and-trade-offs)
10. [Comparison with the Personalized Recommendation Engine](#10-comparison-with-the-personalized-recommendation-engine)
11. [Configuration and Artifact Summary](#11-configuration-and-artifact-summary)

---

## 1. Project Overview

The **Content-Based Similarity Engine** is the component of the system that answers a fundamentally different question from the personalized recommendation engine: given a single movie, what other movies are most *semantically similar* to it?

Unlike the personalized engine, which learns from millions of user interaction histories, this system requires **no user data whatsoever**. It works entirely from the intrinsic content of each film — its plot synopsis, genre tags, thematic keywords, director, and cast. A user inputs one movie title; the engine returns a ranked list of the most similar films in the catalog.

This is achieved through a clean, efficient, and interpretable pipeline:

1. Each movie's content is distilled into a rich natural-language text description.
2. All descriptions are encoded into dense 384-dimensional semantic vectors using a pre-trained Sentence-BERT model.
3. All vectors are L2-normalized at encoding time so that cosine similarity reduces to a plain dot product — the cheapest possible similarity computation.
4. At query time, a single matrix multiplication against the cached embedding matrix scores all 4,803 movies simultaneously.

The result is a system that correctly surfaces sequels for franchise films, finds thematically related films across genres, and handles an optional year-range filter for users who want older or more recent recommendations. The entire inference process involves no learned model weights at query time — all computation is pure linear algebra on pre-computed embeddings.

---

## 2. Dataset

### TMDB 5000 Movies — Two Files

The similarity engine uses the TMDB 5000 dataset exclusively. Unlike the personalized engine (which uses this dataset only as a text enrichment supplement to MovieLens), here it is the **primary and only data source**.

| File | Shape | Key Columns |
|------|-------|-------------|
| `tmdb_5000_movies.csv` | (4803, 20) | `id`, `title`, `overview`, `genres`, `keywords`, `tagline`, `release_date`, `vote_average`, `vote_count`, `popularity`, `budget`, `revenue`, `runtime`, `production_companies` |
| `tmdb_5000_credits.csv` | (4803, 4) | `movie_id`, `title`, `cast`, `crew` |

Both files share the same 4,803 movies. They are joined on the movie ID field to assemble a unified record per film containing both content metadata and cast/crew information.

**Key characteristics of the dataset:**

- Year range: 1916 – 2017 (heavily weighted toward 2000s–2017)
- No interaction data (no user ratings used in this pipeline)
- `genres`, `keywords`, `cast`, and `crew` fields are stored as **JSON strings** (stringified Python lists of dicts), requiring parsing before use
- `vote_count` and `vote_average` fields are used for popularity ranking only, not for similarity computation
- The `id` field from `tmdb_5000_movies.csv` is the TMDB movie ID, which becomes the system's canonical `movie_id`

---

## 3. System Architecture at a Glance

```
TMDB 5000 Movies + Credits
           │
           ▼
  ┌──────────────────────────────────┐
  │   Data Preparation Pipeline     │
  │                                  │
  │  Load & Merge → Parse JSON →    │
  │  Build Description → SBERT →    │
  │  L2-Normalize → Save Artifacts  │
  └──────────────┬───────────────────┘
                 │
         ┌───────▼──────────┐
         │  embeddings.pkl  │  [4803, 384] float32, L2-normalized
         │  metadata.pkl    │  DataFrame, 4803 rows × 9 cols
         └───────┬──────────┘
                 │  (merged into single bundle for production)
         ┌───────▼────────────────────────┐
         │  inference_bundle_tmdb.pkl     │  9.3 MB total
         │  embeddings + metadata + config│
         └───────┬────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────────┐
  │         Query Time (find_similar)           │
  │                                             │
  │  1. Text match query title → get row_idx   │
  │  2. Extract query_vec = embeddings[row_idx] │
  │  3. sims = embeddings @ query_vec.T         │  ← single matmul
  │  4. sims[query_idx] = -1.0                  │  ← exclude self
  │  5. (Optional) year mask applied            │
  │  6. argsort → top-N → format results       │
  └─────────────────────────────────────────────┘
```

---

## 4. Data Preparation Pipeline

**Notebook:** `Data_Prep_similarity.ipynb`

This notebook is entirely self-contained and produces embeddings that require no further training. It is a feature engineering pipeline, not a model training pipeline.

### 4.1 Raw Data Loading and Merging

Both TMDB files are loaded and immediately merged:

```python
credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
df = movies_df.merge(credits_df[['id', 'cast', 'crew']], on='id', how='left')
```

The rename is necessary because the credits file uses `movie_id` while the movies file uses `id` as its primary key — a naming inconsistency in the original TMDB dataset. A left join is used to ensure all 4,803 movies from the movies file are preserved even if a credits row is somehow missing (in practice, all 4,803 rows match, giving a merged shape of (4803, 22)).

### 4.2 JSON Field Parsing

Four columns in the raw TMDB data store structured information as **stringified JSON arrays of dicts** — they look like Python list-of-dict literals but are stored as plain string values in the CSV. These must be parsed before any of their content can be used.

Three utility functions handle this:

**`safe_parse(val)`** — Wraps `ast.literal_eval()` in a try/except, returning an empty list on failure. Handles `NaN` values gracefully. This is more robust than `json.loads()` because TMDB uses single-quoted strings (valid Python, invalid JSON).

**`extract_names(val, key='name', limit=None)`** — Calls `safe_parse` and extracts the value of a specified key from each dict in the resulting list. The `limit` parameter caps the number of items returned.

**`extract_director(crew_val)`** — Iterates through the parsed crew list, returning the `name` field of the first entry whose `job` field equals `"Director"`. Returns an empty string if no director is found.

The following parsed columns are created:

| Column | Source | Content | Limit |
|--------|--------|---------|-------|
| `genres_list` | `genres` | List of genre name strings | None (all) |
| `keywords_list` | `keywords` | List of thematic keyword strings | None (all, up to 15 used in description) |
| `cast_list` | `cast` | List of actor name strings | Top 5 |
| `director` | `crew` | Single director name string | First Director found |

The **cast is capped at 5 actors** — a deliberate choice to balance informativeness against noise. Including 20+ actors would dilute the signal; 5 covers the headline cast who most strongly define a film's identity.

A `year` column is extracted from `release_date` using pandas `dt.year`, and a pipe-delimited `genres_str` column is created for display purposes (e.g., `"Action|Adventure|Fantasy|Science Fiction"`).

**Parsed ranges from the output:**
- Year range: 1916 – 2017
- Total movies after parsing: 4,803 (no rows dropped)

### 4.3 Rich Text Description Construction

The heart of the data preparation. Each movie is converted into a single natural-language text string by the `build_description()` function, which assembles up to five distinct content signals in a fixed priority order:

```
[plot overview]. Keywords: [kw1, kw2, ..., kw15]. Genres: [g1, g2, ...].
Director: [name]. Cast: [actor1, actor2, ..., actor5].
```

**Field-by-field breakdown:**

**1. Plot Overview** (highest priority, placed first)
The `overview` field from TMDB — a multi-sentence natural language synopsis of the film's story. This is the richest semantic signal. It describes what actually happens in the film, the setting, and the central conflict. Being a full paragraph of natural prose, it maps best onto the kinds of sentences that SBERT was pre-trained on. `NaN` values and the literal string `'nan'` are excluded.

**2. Keywords** (up to 15)
TMDB editorial keywords are highly curated thematic tags — things like `"time travel"`, `"based on novel"`, `"dystopia"`, `"space colony"`, `"redemption"`. They are not user-generated tags; they are editorially applied and tend to be precise. Including the top 15 adds thematic density beyond what the plot overview conveys. The `Keywords: ` prefix is prepended to help the language model contextualize these as labels rather than prose.

**3. Genres**
Genre strings like `"Action"`, `"Science Fiction"`, `"Romance"` contribute coarse semantic signal. A film tagged as `"Science Fiction"` will be closer in embedding space to other Science Fiction films purely because that term appears in both descriptions. The `Genres: ` prefix is prepended similarly.

**4. Director**
Director name is a highly predictive signal for stylistic similarity. Two Christopher Nolan films will share `"Director: Christopher Nolan"` in their descriptions. SBERT will learn that names appearing in the same context position across many films signal similarity. This is a deliberate proxy for directing style.

**5. Cast** (top 5 actors)
The five lead actors contribute a complementary signal. Two films starring the same actor (even in different genres) will be pulled slightly closer in embedding space. This is useful for actor-specific discovery ("more films with Cate Blanchett") though it is a softer signal than plot or genre.

**Fallback:** If a movie has no content at all (empty overview, no keywords, no genres), its title is used as the description. In practice, this fallback is never triggered — all 4,803 movies have non-empty descriptions.

**Sample description (Avatar, first movie in dataset):**
```
In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on 
a unique mission, but becomes torn between following orders and protecting an 
alien civilization. Keywords: culture clash, future, space war, space colony, 
society, space travel, futuristic, romance, space, alien, tribe, ...
Genres: Action, Adventure, Fantasy, Science Fiction. 
Director: James Cameron. Cast: Sam Worthington, Zoe Saldana, Sigourney Weaver, 
Stephen Lang, Michelle Rodriguez.
```

### 4.4 SBERT Encoding and L2 Normalization

All 4,803 descriptions are encoded in a single batch pass through `all-MiniLM-L6-v2`:

```python
sbert = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = sbert.encode(
    descriptions,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # ← Critical: L2-normalize at encoding time
)
```

**Why `normalize_embeddings=True` is the most important flag here:**

L2 normalization means every output vector is scaled to have unit length (L2 norm = 1.0). This property means that:

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|) = a · b   (when |a| = |b| = 1)
```

Cosine similarity reduces to a plain dot product. This is not just mathematically convenient — it enables the entire 4,803-movie catalog to be scored with a single matrix multiply:

```python
sims = embeddings @ query_vec.T   # shape: (4803,)
```

One matrix multiply replaces 4,803 individual cosine similarity computations, and requires no division. This is maximally cache-friendly and NumPy-optimized.

The sanity check in the notebook confirms this: `L2 norm of first vec : 1.0000 (should be 1.0)`.

**Output:**
- Shape: `(4803, 384)` — one 384-dim vector per movie
- dtype: `float32` — single precision, sufficient for similarity ranking
- Size on disk: 7.4 MB (embeddings) + 1.9 MB (metadata) = **9.3 MB total bundle**

### 4.5 Metadata Table Construction

A clean metadata DataFrame is assembled for serving results to the API. The columns chosen are exactly what the frontend needs — nothing more:

| Column | Source | Type | Notes |
|--------|--------|------|-------|
| `movie_id` | `df['id']` | int | TMDB movie ID (primary key) |
| `title` | `df['title']` | str | Display title |
| `year` | Parsed from `release_date` | int | 0 if unknown |
| `genres` | `genres_str` | str | Pipe-delimited display string |
| `overview` | `df['overview']` | str | Raw synopsis (stored for potential display) |
| `rating_mean` | `vote_average` | float | TMDB average rating (0–10 scale) |
| `rating_count` | `vote_count` | int | Number of TMDB votes |
| `popularity` | `popularity` | float | TMDB proprietary popularity score |
| `row_idx` | `meta.index` | int | The embedding matrix row for this movie |

The `row_idx` column is the critical link between the metadata table and the embedding matrix. Since the DataFrame is reset-indexed (`meta = meta.reset_index(drop=True)`) before `row_idx` is assigned, `row_idx == DataFrame.index` for every row. This alignment guarantee means `meta.iloc[idx]` and `embeddings[idx]` always refer to the same movie.

**NaN handling:**
- `year` → filled with 0 (used as a sentinel for "unknown year" in output formatting)
- `rating_count` → filled with 0
- `rating_mean` → filled with 0.0
- `popularity` → filled with 0.0

### 4.6 Sanity Checks

Three assertions are run before saving to catch any pipeline errors:

**Check 1 — Row count consistency:**
```python
assert len(embeddings) == len(meta)  # Both must be 4,803
```
This catches any accidental row-dropping between the encoding step and the metadata construction step.

**Check 2 — No NaN in embeddings:**
```python
assert not np.isnan(embeddings).any()
```
A NaN anywhere in the embedding matrix would silently corrupt all similarity scores involving that movie.

**Check 3 — L2 normalization:**
```python
norms = np.linalg.norm(embeddings, axis=1)
assert norms.min() > 0.99 and norms.max() < 1.01
```
Confirms that the `normalize_embeddings=True` flag worked correctly. If norms deviate from 1.0, cosine similarity via dot product would be incorrect, and the entire retrieval math would be wrong.

**Live similarity test:**
The sanity check also runs a live similarity query for "The Matrix" and prints its top-5 results. This serves as a qualitative confirmation that the pipeline is working — if The Matrix's nearest neighbors include other sci-fi action films and its own sequels, the embeddings are semantically meaningful.

Output from sanity check:
```
[OK] Embedding rows match metadata rows : 4,803
[OK] No NaNs in embedding matrix
[OK] All vectors L2-normalized  (min=1.0000, max=1.0000)

Similarity test — movies most similar to "The Matrix":
  0.6816  The Matrix Revolutions (2003)
  0.6608  The Matrix Reloaded (2003)
  0.5604  Commando (1985)
  0.5443  Terminator Genisys (2015)
  0.5419  eXistenZ (1999)
```

### 4.7 Saved Artifacts

**Intermediate artifacts** (saved to `data/working_tmdb/`):

| File | Format | Size | Content |
|------|--------|------|---------|
| `embeddings.pkl` | Pickle | 7.4 MB | `np.ndarray [4803, 384] float32`, L2-normalized |
| `metadata.pkl` | Pickle | 1.9 MB | `pd.DataFrame`, 4803 rows × 9 cols |

**Final inference bundle** (saved to `data/inference_tmdb/`):

| File | Format | Size | Content |
|------|--------|------|---------|
| `inference_bundle_tmdb.pkl` | Pickle (protocol 4) | 9.3 MB | Dict containing embeddings, metadata, `emb_dim=384`, `num_movies=4803` |

The final bundle merges all intermediate artifacts into a single file for clean deployment. Loading one file at server startup is simpler and more robust than loading multiple files and verifying their consistency.

Pickle protocol 4 is used for efficiency — it supports large objects natively and is compatible with Python 3.4+.

---

## 5. The Similarity Engine: Design and Mathematics

**Notebook:** `Model_Training_similarity.ipynb`

This notebook defines the `find_similar()` function, tests it across multiple films, and produces the final inference bundle. There is no gradient computation, no training loop, and no loss function — this is a retrieval system, not a learned model.

### 5.1 Core Principle: Dot Product on Normalized Vectors

The entire retrieval mechanism rests on one mathematical identity:

```
cosine_similarity(a, b) = a · b    iff |a| = |b| = 1.0
```

Because all embedding vectors were L2-normalized during encoding, computing the similarity of a query vector against all 4,803 catalog vectors is simply:

```python
sims = embeddings @ query_vec.T   # (4803, 384) @ (384, 1) → (4803,)
```

This is a matrix-vector product. NumPy executes this via BLAS routines (typically OpenBLAS or MKL on modern hardware), which are highly optimized for exactly this operation. For 4,803 movies with 384-dim vectors, this runs in well under a millisecond on CPU.

The resulting `sims` array contains one cosine similarity score per movie, in the range [-1.0, 1.0]:
- Score of 1.0: identical to the query movie (only the movie itself)
- Score > 0.8: extremely similar (typically sequels or films with nearly identical plots)
- Score 0.5–0.8: meaningfully similar (same franchise/genre/director/themes)
- Score < 0.3: weak or coincidental overlap

### 5.2 Query Movie Lookup (Three-Tier Matching)

Before similarity can be computed, the query string must be resolved to a specific row in the embedding matrix. Title matching is implemented as a three-tier priority system:

**Tier 1 — Exact match** (case-insensitive):
```python
exact = metadata[metadata['title'].str.lower() == q_lower]
```
Whole-title equality, lowercased. "the dark knight" matches "The Dark Knight" exactly.

**Tier 2 — Prefix match** (case-insensitive):
```python
starts = metadata[metadata['title'].str.lower().str.startswith(q_lower)]
```
"the dark" matches "The Dark Knight", "The Dark Knight Rises", "The Dark Hours", etc.

**Tier 3 — Substring match** (case-insensitive):
```python
contains = metadata[metadata['title'].str.lower().str.contains(
    q_lower, regex=False, na=False)]
```
"knight" matches "The Dark Knight", "A Knight's Tale", etc.

The three result sets are concatenated and deduplicated on `movie_id`:
```python
candidates = pd.concat([exact, starts, contains]).drop_duplicates('movie_id')
```

Because exact comes first, then prefix, then contains, and deduplication preserves first occurrence, the most literal match always wins. The first row of `candidates` is selected as the query movie, which will be the best match to the user's intent.

If `candidates` is empty after all three tiers, an error dict is returned:
```python
return {'error': f'No movie found matching "{query}"'}
```

Once the query movie is identified:
```python
query_idx = int(query_row['row_idx'])
query_vec = embeddings[[query_idx]]   # shape: (1, 384) — 2D for matmul
```

The double-bracket indexing `[[query_idx]]` is intentional — it keeps the result as a 2D array `(1, 384)` rather than a 1D vector `(384,)`, which is required for the matrix multiply to produce the correct output shape.

### 5.3 Similarity Scoring Against the Full Catalog

```python
sims = (embeddings @ query_vec.T).squeeze()   # (4803, 384) @ (384, 1) → squeeze to (4803,)
sims[query_idx] = -1.0                         # Exclude the query movie itself
```

Setting the query movie's own score to -1.0 is the cleanest way to exclude it from results. Since all real similarity scores are between -1.0 and 1.0, and the query movie's true score would be 1.0 (identical to itself), setting it to -1.0 pushes it to the absolute bottom of any ranking.

### 5.4 Year Range Filtering with Score Masking

An optional year range filter allows users to constrain results to a specific era (e.g., only films from 2000 onwards, or only classic films before 1990).

The filter is implemented through **score masking** rather than DataFrame filtering. This is a key design choice — it operates entirely on NumPy arrays without any pandas row-dropping:

```python
if year_from is not None or year_to is not None:
    years = metadata['year'].values       # numpy array of int years
    mask  = np.ones(len(sims), dtype=np.float32)

    if year_from is not None:
        mask[years < year_from] = 0.0
    if year_to is not None:
        mask[years > year_to]   = 0.0

    sims = sims * mask - 2.0 * (1.0 - mask)
```

**How the masking math works:**

For any movie at index `i`:
- **In-range movie** (`mask[i] = 1.0`): `sims[i] * 1.0 - 2.0 * 0.0 = sims[i]` — score is unchanged
- **Out-of-range movie** (`mask[i] = 0.0`): `sims[i] * 0.0 - 2.0 * 1.0 = -2.0` — score is set to -2.0

Since all real similarity scores fall in [-1.0, 1.0], setting out-of-range movies to -2.0 mathematically guarantees they will rank below all in-range movies, even if those in-range movies have very low similarity. Out-of-range movies are effectively excluded without requiring any conditional branching or array slicing.

This is a numerically clean, vectorized, branch-free approach — the entire operation is two element-wise multiplications and one subtraction, all handled by NumPy in a single pass.

### 5.5 Top-N Retrieval and Result Formatting

```python
top_idxs = sims.argsort()[::-1][:top_n]
```

`argsort()` returns indices that would sort the array in ascending order. `[::-1]` reverses this to descending. `[:top_n]` takes the first `top_n` elements — the indices of the most similar movies.

Each result is formatted as a dict with five fields:

```python
{
    'movie_id'  : int(row['movie_id']),         # TMDB movie ID
    'title'     : str(row['title']),            # Display title
    'genres'    : str(row['genres']),           # Pipe-delimited genre string
    'year'      : int(row['year']) if row['year'] > 0 else None,  # None if unknown
    'similarity': round(float(sims[idx]), 4),  # Cosine similarity, 4 decimal places
}
```

The `similarity` score is included in the output — unlike the personalized engine's dummy `predicted_rating`, this is a **real, interpretable number**. A score of 0.86 between "The Dark Knight" and "The Dark Knight Rises" is a meaningful, mathematically grounded statement that those two films are highly semantically similar.

The complete return structure distinguishes the query movie from the results:

```python
{
    'query': {
        'movie_id': ..., 'title': ..., 'genres': ..., 'year': ...
    },
    'similar': [result_1, result_2, ..., result_top_n]
}
```

This separation allows the frontend to display "Because you selected: [query movie]" and "Similar movies:" in distinct UI sections.

---

## 6. Production Inference Module

The production inference code (deployed inside Django) is a stripped-down, dependency-minimal version of the notebook's functions. It removes the year-filter functionality (a UI decision — the production frontend doesn't expose year range controls) and tightens the implementation.

### 6.1 Lazy Singleton Loading

```python
_bundle = None

def _load():
    global _bundle
    if _bundle is not None:
        return
    with open(settings.INFERENCE_BUNDLE_PATH_TMDB, 'rb') as f:
        _bundle = pickle.load(f)
```

The bundle is loaded at most once per server process. The path is stored in Django settings (`INFERENCE_BUNDLE_PATH_TMDB`), keeping the path configuration separate from the inference logic. At 9.3 MB, the bundle loads extremely quickly (sub-second on any modern system), making the lazy load overhead negligible.

Unlike the personalized engine's two-model loading (which required separate stage-1 and stage-2 loaders with different initialization logic), the similarity engine's loader is a single trivial function — load a file, store it in a global. No PyTorch, no GPU, no model instantiation.

### 6.2 The Inference Bundle

The bundle is a Python dict containing:

| Key | Type | Value |
|-----|------|-------|
| `'embeddings'` | `np.ndarray` | `(4803, 384) float32`, L2-normalized |
| `'metadata'` | `pd.DataFrame` | 4,803 rows × 9 columns |
| `'emb_dim'` | `int` | `384` |
| `'num_movies'` | `int` | `4803` |

The `emb_dim` and `num_movies` keys serve as self-documentation — any code loading this bundle can verify its dimensions without inspecting the arrays themselves.

### 6.3 find_similar() — Production Version

The production version is a faithful simplification of the notebook version:

```python
def find_similar(query, top_n=20):
    _load()
    meta  = _bundle['metadata']
    embs  = _bundle['embeddings']

    # Three-tier title match
    q_low    = query.lower().strip()
    exact    = meta[meta['title'].str.lower() == q_low]
    starts   = meta[meta['title'].str.lower().str.startswith(q_low)]
    contains = meta[meta['title'].str.lower().str.contains(q_low, regex=False, na=False)]
    cands    = pd.concat([exact, starts, contains]).drop_duplicates('movie_id')

    if cands.empty:
        return {'error': f'No movie found matching "{query}"'}

    qrow = cands.iloc[0]
    qidx = int(qrow['row_idx'])
    qvec = embs[[qidx]]               # (1, 384)

    sims          = (embs @ qvec.T).squeeze()   # (4803,)
    sims[qidx]    = -1.0              # exclude self

    top_idxs = sims.argsort()[::-1][:top_n]

    results = []
    for idx in top_idxs:
        r = meta.iloc[idx]
        results.append({
            'movie_id'  : int(r.movie_id),
            'title'     : str(r.title),
            'genres'    : str(r.genres),
            'year'      : int(r.year) if r.year > 0 else None,
            'similarity': round(float(sims[idx]), 4),
        })

    return {
        'query'  : {'movie_id': int(qrow.movie_id), 'title': str(qrow.title),
                    'genres': str(qrow.genres)},
        'similar': results,
    }
```

**Differences from the notebook version:**
- No `year_from` / `year_to` parameters (removed from production API)
- Accesses `meta` and `embs` from the loaded `_bundle` dict rather than module-level globals
- `_load()` call at function entry ensures the bundle is available

### 6.4 search_movies() — Title Search

```python
def search_movies(q, limit=15):
    _load()
    meta    = _bundle['metadata']
    q_lower = q.lower().strip()

    starts   = meta[meta['title'].str.lower().str.startswith(q_lower)]
    contains = meta[~meta['title'].str.lower().str.startswith(q_lower) &
                    meta['title'].str.lower().str.contains(q_lower, regex=False, na=False)]

    combined = pd.concat([starts, contains]).drop_duplicates('movie_id').head(limit)

    return [{'movie_id': int(r.movie_id), 'title': str(r.title),
             'year': int(r.year) if r.year > 0 else None,
             'genres': str(r.genres)}
            for _, r in combined.iterrows()]
```

This is the typeahead/autocomplete function. It takes a partial query string and returns up to 15 matching movies in prefix-first order. The `~startswith` clause in the contains filter explicitly excludes prefix matches to prevent duplication — prefix matches come first in the concatenated result, and `~startswith` ensures the `contains` group only contributes genuinely new matches.

There is no error case — if no titles match, an empty list is returned, which is the appropriate behavior for an autocomplete widget.

---

## 7. Supporting Utilities

### 7.1 Popular Movies

```python
def get_popular_tmdb(n: int = 50) -> list:
    top = (metadata.sort_values('rating_count', ascending=False)
                   .head(n)
                   .reset_index(drop=True))
    ...
```

Sorts the metadata table by `rating_count` (TMDB vote count, which reflects real-world popularity) and returns the top-N movies. This is used for the initial state of the similarity page — before the user has typed anything, the UI shows a curated list of well-known films to query against.

Sorting by `vote_count` (number of votes) rather than `vote_average` (average rating) is intentional. Vote count reflects cultural reach and familiarity — films like Inception (13,752 votes) and The Dark Knight (12,002 votes) are highly recognizable anchors. Sorting by rating average alone would surface obscure films with high scores from few reviewers, which would be poor defaults for a discovery interface.

**Top 10 popular movies by vote count:**

| Rank | Title | Year | Vote Count | Avg Rating |
|------|-------|------|-----------|-----------|
| 1 | Inception | 2010 | 13,752 | 8.1 |
| 2 | The Dark Knight | 2008 | 12,002 | 8.2 |
| 3 | Avatar | 2009 | 11,800 | 7.2 |
| 4 | The Avengers | 2012 | 11,776 | 7.4 |
| 5 | Deadpool | 2016 | 10,995 | 7.4 |
| 6 | Interstellar | 2014 | 10,867 | 8.1 |
| 7 | Django Unchained | 2012 | 10,099 | 7.8 |
| 8 | Guardians of the Galaxy | 2014 | 9,742 | 7.9 |
| 9 | The Hunger Games | 2012 | 9,455 | 6.9 |
| 10 | Mad Max: Fury Road | 2015 | 9,427 | 7.2 |

### 7.2 TMDB Search (Notebook Version)

```python
def search_movies_tmdb(q: str, limit: int = 15) -> list
```

The notebook implements and tests its own version of the title search function under the name `search_movies_tmdb`. It is functionally identical to the production `search_movies()` — same two-pass prefix-then-contains logic, same deduplication, same output format. It exists separately in the notebook for standalone testing without a Django environment.

Tested queries from the notebook:
```
search_movies_tmdb("inter") →
  Interstellar (2014)
  Interview with the Vampire (1994)
  Interview with the Assassin (2002)
  Captain America: The Winter Soldier (2014)
  The Huntsman: Winter's War (2016)

search_movies_tmdb("the dark") →
  The Dark Knight Rises (2012)
  The Dark Knight (2008)
  The Darkest Hour (2011)
  The Dark Hours (2005)
  Thor: The Dark World (2013)
```

This confirms that prefix matching correctly surfaces "Interstellar" as the first result for "inter" despite "Interview" and "Winter" both containing the substring.

---

## 8. Empirical Similarity Results

The notebook includes a structured test suite across five representative films. These results provide qualitative evidence of what the embedding space captures:

### The Dark Knight (2008) — Drama|Action|Crime|Thriller

| Similarity | Movie |
|-----------|-------|
| 0.8587 | The Dark Knight Rises (2012) |
| 0.7697 | Batman (1989) |
| 0.7387 | Batman v Superman: Dawn of Justice (2016) |
| 0.7382 | Batman Begins (2005) |
| 0.7313 | Batman Forever (1995) |

**Interpretation:** Near-perfect franchise detection. All five results are Batman films. The highest score (0.86) is to The Dark Knight Rises — same director, same cast, direct narrative continuation. The scores decay gracefully as the films become more stylistically distant (Batman 1989 is a different era/tone but same franchise).

### Toy Story (1995) — Animation|Comedy|Family

| Similarity | Movie |
|-----------|-------|
| 0.8051 | Toy Story 2 (1999) |
| 0.7336 | Toy Story 3 (2010) |
| 0.5240 | Big (1988) |
| 0.5127 | St. Vincent (2014) |
| 0.5094 | Ted 2 (2015) |

**Interpretation:** The two sequels are correctly identified at high similarity (0.80 and 0.73). The remaining results cluster around films involving themes of childhood, toys coming alive, and the relationship between children and adults — thematically coherent even if not the same franchise.

### Inception (2010) — Action|Thriller|Science Fiction|Mystery|Adventure

| Similarity | Movie |
|-----------|-------|
| 0.5720 | Memento (2000) |
| 0.5601 | Kiss Kiss Bang Bang (2005) |
| 0.5543 | The Spanish Prisoner (1997) |
| 0.5520 | Identity Thief (2013) |
| 0.5479 | Hearts in Atlantis (2001) |

**Interpretation:** Inception's lower similarity scores (max 0.57 vs Dark Knight's 0.86) reflect its highly distinctive and original narrative structure — there are few films in the TMDB 5000 catalog that genuinely resemble its dream-heist premise. Memento (also Christopher Nolan, also mind-bending and non-linear) correctly surfaces as the most similar. The results cluster around psychological thrillers and films involving deception and altered reality.

### The Matrix (1999) — Action|Science Fiction — Year Filter: ≥ 2000

| Similarity | Movie |
|-----------|-------|
| 0.6816 | The Matrix Revolutions (2003) |
| 0.6608 | The Matrix Reloaded (2003) |
| 0.5443 | Terminator Genisys (2015) |
| 0.5388 | Transcendence (2014) |
| 0.5294 | Surrogates (2009) |

**Interpretation:** With the year filter active (≥ 2000), the 1999 Matrix itself and any pre-2000 films are excluded. The sequels (2003) correctly dominate. The remaining results are science-fiction action films dealing with AI, machine uprising, and post-human themes — thematically tight.

### Interstellar (2014) — Adventure|Drama|Science Fiction

| Similarity | Movie |
|-----------|-------|
| 0.6417 | Prometheus (2012) |
| 0.5723 | Gattaca (1997) |
| 0.5457 | Close Encounters of the Third Kind (1977) |
| 0.5403 | The Black Hole (1979) |
| 0.5305 | Seeking a Friend for the End of the World (2012) |

**Interpretation:** No direct sequels exist, so the model surfaces thematically related science fiction films dealing with space exploration, first contact, and humanity's place in the cosmos. Prometheus (space exploration, alien discovery) and Gattaca (genetic destiny, humanity's future) are apt. The breadth of years (1977–2012) demonstrates the system's ability to surface classics alongside contemporary films based purely on semantic content.

---

## 9. Key Design Decisions and Trade-offs

### No Machine Learning at Query Time

This is a deliberate architectural choice. The entire "model" is the embedding matrix — a static artifact produced once by a pre-trained encoder. Query-time inference is pure linear algebra: one matrix multiply, one argsort. This makes the system:

- **Deterministic**: The same query always returns the same results.
- **Stateless**: No model state to maintain or reload between requests.
- **Extremely fast**: Sub-millisecond for the core similarity computation.
- **Interpretable**: The similarity score is a real, meaningful number (cosine similarity).

The trade-off is that the system cannot learn from user feedback. If users consistently click on results that are ranked low, the system has no mechanism to adjust. This is acceptable for a content-based engine where "objectively similar" is the design goal.

### Why SBERT Instead of a Trained Neural Collaborative Filter?

Collaborative filtering (as used in the personalized engine) would require user-item interaction data to determine item-item similarity. Two films would be "similar" because the same users rated both highly. This can surface unexpected but legitimate connections (users who love film X also love film Y, even if the plots are nothing alike).

Content-based similarity answers a different question: are these films *objectively similar in content*? SBERT on rich text descriptions captures this directly. It requires no interaction data, works for cold-start items with zero ratings, and produces interpretable similarity reasons (two films are similar because they share a plot about space exploration, a director, and the same keywords).

### Why Include Director and Cast in the Description?

The alternative would be to embed only the plot overview. However:
- Two films by the same director are often stylistically similar in ways not captured by plot text (pacing, cinematography style, thematic obsessions) — including the director name provides a proxy signal.
- Franchise films often share cast members across sequels. Including cast reinforces these connections.
- The risk is over-indexing on shared actors (two unrelated films both starring Tom Hanks might be pulled artificially close). Capping cast at 5 and weighting it as the last component in the description string mitigates this.

### Why Not Weighted Concatenation?

The description string treats all components as natural language, with no explicit weighting. A potential improvement would be to repeat high-value fields (like overview) or use a weighted sum of separate embeddings for each field. The current approach is simpler, transparent, and effective — the test results confirm that the ordering heuristic (overview first, richest signal) combined with SBERT's attention mechanism produces good results without explicit weights.

### Why 4,803 Movies Instead of the Full 25M MovieLens Catalog?

The similarity engine uses the TMDB 5000 dataset exclusively — not the full MovieLens catalog — because:

1. **Text quality**: TMDB 5000 has clean, editorially verified metadata. Not all MovieLens movies have corresponding TMDB data.
2. **Scope**: The TMDB 5000 covers the most culturally significant films. A user asking for movies similar to "Inception" benefits more from a high-quality 5K catalog than a noisy 60K catalog where many entries have sparse or missing metadata.
3. **Efficiency**: 4,803 × 384 float32 = 7.4 MB. The full MovieLens similarity matrix at 60K movies would be 92 MB — not infeasible, but requiring a meaningfully different infrastructure approach.

The trade-off is that the similarity engine cannot surface recommendations from the long tail of obscure films that exist in MovieLens but not in TMDB 5000.

---

## 10. Comparison with the Personalized Recommendation Engine

Understanding both systems together illuminates why each exists and where each one shines.

| Dimension | Personalized Engine | Similarity Engine |
|-----------|--------------------|--------------------|
| **Input** | 15 rated movies | 1 movie title |
| **What it answers** | "What should *I* watch next?" | "What is *similar* to this movie?" |
| **User data required** | Yes — 15 ratings mandatory | None — fully cold-start capable |
| **Catalog size** | ~10,321 movies (filtered MovieLens) | 4,803 movies (TMDB 5000) |
| **Core models** | Two-Tower + SASRec (trained neural nets) | SBERT embeddings (pre-trained, no fine-tuning) |
| **Similarity basis** | User behavior patterns | Movie content (plot, genre, director, cast) |
| **Output score** | Dummy rank score for frontend ordering | Real cosine similarity (interpretable) |
| **Inference compute** | Two model forward passes + matmul | One matmul |
| **Determinism** | Deterministic (no sampling) | Deterministic |
| **Framework** | PyTorch | NumPy only |
| **GPU benefit** | Significant (Two-Tower + SASRec forward passes) | Negligible (matmul on 4803 × 384 is trivially fast on CPU) |
| **Bundle size** | ~200–500 MB (model weights + embeddings) | 9.3 MB |
| **Cold-start capable** | No (needs rated movies) | Yes (works for any movie in catalog) |
| **Handles "more like this"** | Indirectly (via history) | Directly and precisely |
| **Handles "what's next for me"** | Directly and precisely | Not designed for this |

The two engines are complementary, not competing. A user who rates 15 movies gets personalized recommendations from the full two-stage hybrid pipeline. A user who spots a specific film they like on the popular movies list can immediately ask "what's similar to this?" and get content-based results — no rating history required.

---

## 11. Configuration and Artifact Summary

### Pipeline Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset | TMDB 5000 Movies + Credits | Two CSV files |
| Total movies | 4,803 | No filtering applied |
| Text encoder | `all-MiniLM-L6-v2` | Via `sentence-transformers` library |
| Embedding dimension | 384 | Fixed by model architecture |
| Normalization | L2 (unit vectors) | Applied at encoding time |
| Encoding batch size | 128 | Progress bar enabled |
| Output dtype | `float32` | Single precision, sufficient for ranking |
| Pickle protocol | 4 | Python 3.4+ compatible, large-object efficient |
| Similarity metric | Cosine (via dot product) | Exact, enabled by L2 normalization |
| Default top-N | 20 | Configurable per call |
| Year filter | Optional | Implemented via score masking (not row-drop) |
| Cast limit | 5 actors | Balance of signal vs. noise |
| Keyword limit | 15 keywords | In description string |

### Artifact Inventory

| Artifact | Location | Size | Description |
|----------|----------|------|-------------|
| `embeddings.pkl` | `data/working_tmdb/` | 7.4 MB | `np.ndarray [4803, 384] float32` |
| `metadata.pkl` | `data/working_tmdb/` | 1.9 MB | `pd.DataFrame`, 9 columns |
| `inference_bundle_tmdb.pkl` | `data/inference_tmdb/` | 9.3 MB | Merged production bundle |

### Metadata Schema

| Column | Type | Description |
|--------|------|-------------|
| `movie_id` | int | TMDB movie ID (primary key) |
| `title` | str | Display title |
| `year` | int | Release year (0 if unknown) |
| `genres` | str | Pipe-delimited genre string |
| `overview` | str | Raw TMDB plot synopsis |
| `rating_mean` | float | TMDB average rating (0–10) |
| `rating_count` | int | TMDB vote count |
| `popularity` | float | TMDB proprietary popularity score |
| `row_idx` | int | Index into embedding matrix (== DataFrame index) |

---

*Report prepared from source notebooks: `Data_Prep_similarity.ipynb`, `Model_Training_similarity.ipynb`, and the production inference module.*
