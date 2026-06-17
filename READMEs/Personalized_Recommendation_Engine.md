# Personalized Movie Recommendation Engine: A Hybrid Deep Learning System

## Technical Report — Two-Tower Candidate Generation with SASRec Sequential Re-Ranking

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Datasets](#2-datasets)
3. [System Architecture at a Glance](#3-system-architecture-at-a-glance)
4. [Data Preparation: Two-Tower Pipeline](#4-data-preparation-two-tower-pipeline)
   - 4.1 [Raw Data Loading](#41-raw-data-loading)
   - 4.2 [Filtering and Implicit Feedback Construction](#42-filtering-and-implicit-feedback-construction)
   - 4.3 [ID Remapping](#43-id-remapping)
   - 4.4 [TMDB Enrichment and Text Feature Engineering](#44-tmdb-enrichment-and-text-feature-engineering)
   - 4.5 [Sentence Embedding Generation](#45-sentence-embedding-generation)
   - 4.6 [Saved Artifacts](#46-saved-artifacts)
5. [Data Preparation: Sequential Pipeline](#5-data-preparation-sequential-pipeline)
   - 5.1 [Loading and Column Normalization](#51-loading-and-column-normalization)
   - 5.2 [Filtering and Contiguous Index Mapping](#52-filtering-and-contiguous-index-mapping)
   - 5.3 [Sequence Construction and Padding Strategy](#53-sequence-construction-and-padding-strategy)
   - 5.4 [Leave-One-Out Train/Val/Test Split](#54-leave-one-out-trainvaltest-split)
   - 5.5 [Saved Artifacts](#55-saved-artifacts)
6. [Model Architecture: History-Based Two-Tower Network](#6-model-architecture-history-based-two-tower-network)
   - 6.1 [Design Philosophy](#61-design-philosophy)
   - 6.2 [Components](#62-components)
   - 6.3 [Forward Pass and Training Objective](#63-forward-pass-and-training-objective)
   - 6.4 [Inference: Cached Item Representations](#64-inference-cached-item-representations)
7. [Model Architecture: SASRec (Self-Attentive Sequential Recommender)](#7-model-architecture-sasrec-self-attentive-sequential-recommender)
   - 7.1 [Design Philosophy](#71-design-philosophy)
   - 7.2 [Components](#72-components)
   - 7.3 [Causal Masking](#73-causal-masking)
   - 7.4 [Output Head and Prediction](#74-output-head-and-prediction)
8. [The Two-Stage Recommendation Pipeline (Inference)](#8-the-two-stage-recommendation-pipeline-inference)
   - 8.1 [Input Contract](#81-input-contract)
   - 8.2 [Stage 1: Candidate Generation (Two-Tower)](#82-stage-1-candidate-generation-two-tower)
   - 8.3 [Stage 2: Sequential Intent Scoring (SASRec)](#83-stage-2-sequential-intent-scoring-sasrec)
   - 8.4 [Blending: Sequel Boost → SASRec Top 5 → Two-Tower Fill](#84-blending-sequel-boost--sasrec-top-5--two-tower-fill)
   - 8.5 [Output Formatting](#85-output-formatting)
9. [Model Loading: Lazy Singleton Pattern](#9-model-loading-lazy-singleton-pattern)
10. [Supporting Utilities](#10-supporting-utilities)
    - 10.1 [Popular Movies](#101-popular-movies)
    - 10.2 [Search](#102-search)
    - 10.3 [Franchise Detection Heuristic](#103-franchise-detection-heuristic)
11. [Key Design Decisions and Trade-offs](#11-key-design-decisions-and-trade-offs)
12. [Complete Data and Model Flow Diagram (Textual)](#12-complete-data-and-model-flow-diagram-textual)
13. [Hyperparameters and Configuration Summary](#13-hyperparameters-and-configuration-summary)

---

## 1. Project Overview

This report documents the **Personalized Recommendation Engine** — the component of the system that takes a user's 15 explicit movie ratings and returns a ranked list of personalized movie recommendations. The system does not use a traditional single-model approach. Instead, it implements an industry-standard **two-stage retrieval-and-ranking pipeline** that combines two complementary deep learning models:

- **Stage 1 — History-Based Two-Tower Network**: A content-aware collaborative filtering model that understands a user's *general, long-term taste* based on their entire rating history. It rapidly narrows the full catalog (~10,000+ movies) down to a manageable pool of high-quality candidates.

- **Stage 2 — SASRec (Self-Attentive Sequential Recommender)**: A Transformer-based sequential model that captures the user's *short-term, immediate intent* by analyzing the temporal order of their most recent 5-star interactions. It re-ranks and supplements the Stage 1 pool to surface what the user most likely wants *right now*.

The final output blends signals from both models plus a **franchise/sequel boosting heuristic**, producing a ranked list of up to 20 recommendations that is simultaneously broad (long-term preferences) and contextually sharp (current mood and intent).

---

## 2. Datasets

The system is trained and evaluated on two public datasets:

### MovieLens 25M (`ml-25m`)

The primary interaction dataset. Contains approximately **25 million explicit ratings** from real users of the MovieLens platform.

| File | Content |
|------|---------|
| `ratings.csv` | `userId`, `movieId`, `rating` (0.5–5.0 stars), `timestamp` |
| `movies.csv` | `movieId`, `title`, `genres` (pipe-separated) |
| `links.csv` | `movieId`, `imdbId`, `tmdbId` — cross-reference table |

Raw dataset size: 25,000,095 ratings across 162,541 users and 59,047 movies.

### TMDB 5000 Movies (`tmdb_5000_movies.csv`)

A supplementary metadata dataset from The Movie Database. Used exclusively to enrich movies with:

- `overview`: A long-form textual synopsis of the film's plot.
- `tagline`: A short marketing tagline.
- `id` (aliased to `tmdbId`): The primary key used to join with MovieLens via the `links.csv` cross-reference.

These text fields are the raw material for generating semantic movie embeddings.

---

## 3. System Architecture at a Glance

```
User Input: 15 Movie Ratings
           │
           ├──► [Stage 1] Two-Tower Network
           │         │ Uses liked items as user proxy
           │         │ Scores all ~10K items via dot product
           │         │ Returns top-100 candidates
           │
           ├──► [Stage 2] SASRec Network
           │         │ Uses last five 5-star movies in sequence
           │         │ Predicts next-item logits over 59K items
           │         │ Returns top-5 sequentially-relevant picks
           │
           └──► Blending Logic
                    │ 1. Franchise/Sequel Boost (from last 5-star)
                    │ 2. SASRec Top-5 inserted first
                    │ 3. Two-Tower candidates fill remainder
                    │
                    ▼
              Final Ranked List (top-20 movies)
```

---

## 4. Data Preparation: Two-Tower Pipeline

**Notebook:** `Data_Prep_2Tower.ipynb`

This notebook processes raw MovieLens and TMDB data into clean training interaction files and pre-computed semantic embeddings for every movie.

### 4.1 Raw Data Loading

Three MovieLens files are loaded: `ratings.csv`, `movies.csv`, and `links.csv`. The TMDB enrichment file `tmdb_5000_movies.csv` is loaded from a separate path. All loading is done with pandas.

### 4.2 Filtering and Implicit Feedback Construction

The raw 25M ratings are aggressively filtered to ensure data quality:

**Step 1 — Minimum movie popularity filter:**
Only movies with **≥ 100 ratings** are retained. This eliminates long-tail, obscure movies with insufficient interaction signal, reducing noise in collaborative filtering.

**Step 2 — Minimum user activity filter:**
Only users with **≥ 50 ratings** are retained. This ensures every user has a meaningful interaction history for the model to learn from. Cold-start users (few interactions) are excluded from training.

After these two filters:

| Metric | Value |
|--------|-------|
| Positive interactions | 11,158,595 |
| Unique users | 102,128 |
| Unique movies | 10,326 |

**Step 3 — Implicit feedback binarization:**
The system treats recommendations as a **implicit feedback problem**, not an explicit rating prediction problem. A rating of **≥ 4.0 stars is treated as a "like"** (positive signal). All lower ratings are discarded. This produces a clean set of positive interactions without any negative explicit ratings — the model learns from what users *chose to like*, not from predicted star counts.

Only the columns `userId`, `movieId`, and `timestamp` are retained from this point forward.

### 4.3 ID Remapping

Raw MovieLens IDs are non-contiguous integers (e.g., movieId can be 1, 5, 147, 318...). Neural embedding layers require contiguous 0-indexed integer indices. Two mapping dictionaries are created:

```python
user2idx = {original_user_id: contiguous_int}   # 102,128 entries
movie2idx = {original_movie_id: contiguous_int}  # 10,326 entries
```

Both the positive interactions DataFrame and the movie metadata DataFrame are augmented with `user_idx` and `movie_idx` columns using these maps. These dictionaries are saved and are essential for inference — they translate between real-world MovieLens IDs and the model's internal index space.

### 4.4 TMDB Enrichment and Text Feature Engineering

The movie metadata is enriched through a multi-step join:

1. `movies_filtered` (MovieLens titles + genres) is joined with `links.csv` to obtain the `tmdbId` for each movie.
2. Rows missing a `tmdbId` are dropped (a small subset of movies with no TMDB match).
3. The result is joined with the TMDB dataset on `tmdbId` to obtain `overview` and `tagline` fields.
4. Missing text fields are filled with empty strings.

A **rich text feature string** is then constructed for each movie by concatenating three fields:

```
text_feature = title + ". " + tagline + " " + overview
```

This concatenation deliberately puts the movie title first (highest-signal term for retrieval), followed by the punchy tagline, followed by the detailed plot overview. The resulting string gives the sentence encoder a dense, information-rich passage to embed.

After the TMDB join, 5 movies that had no TMDB match are lost, reducing the movie count from 10,326 to **10,321** and removing their interactions. Final cleaned interactions: **11,158,008**.

### 4.5 Sentence Embedding Generation

The `all-MiniLM-L6-v2` model from the `sentence-transformers` library is used to encode every movie's `text_feature` string into a dense semantic vector.

**Why `all-MiniLM-L6-v2`?**
- Output dimensionality: **384 dimensions**
- It is a distilled, lightweight model that runs very fast (particularly important when embedding 10K+ items)
- Despite its small size, it captures rich semantic relationships — two movies about "space colonization" will have similar embeddings even if they share no exact vocabulary
- The model is pre-trained on a massive diverse corpus; its semantic knowledge generalizes well to movie plots

Encoding is performed with `batch_size=256` and a progress bar. The output is a dictionary mapping each `movie_idx` to its 384-dimensional float32 embedding vector.

### 4.6 Saved Artifacts

The following files are saved to `data/processed/`:

| File | Format | Description |
|------|--------|-------------|
| `train_interactions.csv` | CSV | 80% random split of positive interactions (user_idx, movie_idx, timestamp) |
| `val_interactions.csv` | CSV | 20% random split for validation |
| `user2idx.pkl` | Pickle | User ID → contiguous index map |
| `movie2idx.pkl` | Pickle | Movie ID → contiguous index map |
| `movie_embeddings.pkl` | Pickle | Dict: movie_idx → 384-dim numpy array |
| `movie_metadata.csv` | CSV | movie_idx, movieId, title, genres per movie |

The train/val split uses `sklearn`'s `train_test_split` with `random_state=42` and `test_size=0.2`. Note: this is a **random** split at the interaction level (not a user-level or time-based split), which is appropriate for the Two-Tower model's training objective.

---

## 5. Data Preparation: Sequential Pipeline

**Notebook:** `Data_Prep_Sequential.ipynb`

This notebook processes the same raw MovieLens data through a fundamentally different pipeline optimized for sequence modeling. The sequential model needs to see *ordered interaction histories*, not just unordered bags of liked items.

### 5.1 Loading and Column Normalization

The raw `ratings.csv` is loaded and columns are renamed to standard names (`userId` → `user_id`, `movieId` → `item_id`) for clarity. All 25,000,095 interactions are loaded — unlike the Two-Tower pipeline, the sequential model uses **all ratings** (not just positives ≥ 4.0), because what a user rated at all (regardless of score) represents a meaningful temporal signal about their browsing behavior.

### 5.2 Filtering and Contiguous Index Mapping

**Minimum interaction filter:**
Users with fewer than `MIN_INTERACTIONS = 5` total ratings are removed. This ensures every user has at least a minimal sequence to learn from. After filtering: 162,541 unique users, 59,047 unique items.

**Contiguous index mapping (1-indexed):**
Unlike the Two-Tower pipeline which uses 0-indexed mappings, the sequential pipeline uses **1-indexed** mappings:

```python
user2idx = {u: i + 1 for i, u in enumerate(user_ids)}
item2idx = {i: j + 1 for j, i in enumerate(item_ids)}
```

Index 0 is reserved as the **padding token**. This is a standard design pattern for sequential recommendation — when a user's history is shorter than the maximum sequence length, zeros are prepended to bring it to full length. The embedding layer is initialized with `padding_idx=0`, meaning the padding token always produces a zero vector and does not contribute to learning.

### 5.3 Sequence Construction and Padding Strategy

Data is sorted chronologically per user (`sort_values(['user_id', 'timestamp'])`). Each user's items are then grouped into a chronological list.

A sliding window is applied over each user's sequence to generate training samples. For position `i` in a user's sequence:

- **Target:** `seq[i]` — the item the user interacted with at step `i`
- **Context:** `seq[max(0, i-20):i]` — up to the last 20 items before this interaction

**Left (pre) padding:**
If the context window has fewer than `MAX_SEQ_LENGTH = 20` items, it is left-padded with zeros to reach length 20:

```python
pad_len = MAX_SEQ_LENGTH - len(context)
padded_context = [0] * pad_len + context
```

This means a user's very first interaction gets a context of 20 zeros, and a user with 5 previous interactions gets 15 zeros followed by 5 real items. This is **native pre-padding**, meaning the model sees real items at the *right* end of the sequence, which aligns with the positional embeddings and causal attention mask used in SASRec.

### 5.4 Leave-One-Out Train/Val/Test Split

A standard leave-one-out split strategy is applied *per user*:

| Split | Definition |
|-------|-----------|
| **Test** | The very last interaction of each user (`i == len(seq) - 1`) |
| **Validation** | The second-to-last interaction of each user (`i == len(seq) - 2`) |
| **Train** | All other interactions |

This strategy ensures:
- Every user appears in all three splits
- Validation and test examples reflect real "next-item" prediction scenarios
- There is no future leakage into training

Final split sizes:

| Split | Samples |
|-------|---------|
| Train | 24,512,472 |
| Validation | 162,541 |
| Test | 162,541 |

### 5.5 Saved Artifacts

All tensors are saved as PyTorch `.pt` files for efficient loading during training:

| File | Type | Shape |
|------|------|-------|
| `train_data.pt` | Dict with `X` (LongTensor) and `y` (LongTensor) | X: (24.5M, 20), y: (24.5M,) |
| `val_data.pt` | Same structure | X: (162K, 20), y: (162K,) |
| `test_data.pt` | Same structure | X: (162K, 20), y: (162K,) |
| `mappings.pkl` | Pickle dict | `{'user2idx': ..., 'item2idx': ...}` |

Using `dtype=torch.long` is critical — PyTorch's `nn.Embedding` requires integer (Long) indices as input.

---

## 6. Model Architecture: History-Based Two-Tower Network

**Class:** `HistoryBasedTwoTower`

### 6.1 Design Philosophy

The Two-Tower (also called Dual Encoder) architecture is an industry-standard approach for large-scale retrieval used at companies like Google, YouTube, and Spotify. The core idea is to separately learn a **user embedding** and an **item embedding** in the same vector space, then score user-item affinity via dot product similarity.

In this implementation, there is no dedicated user embedding table. Instead, the **user is represented as the mean of the embeddings of all items they have liked.** This has several advantages:

- It naturally handles new users whose IDs were never seen during training (any liked items can form a user vector)
- It keeps the model lightweight — no user embedding matrix of size `(102K × 64)` needs to be stored and loaded
- It makes inference dynamic: the user representation is computed on-the-fly at query time from whatever items the user rates highly

### 6.2 Components

**a) Frozen Text Embeddings (`nn.Embedding.from_pretrained`, frozen)**

A lookup table initialized from the pre-computed `all-MiniLM-L6-v2` sentence embeddings (384 dimensions per movie). This layer is set to `freeze=True`, meaning its weights are not updated during backpropagation. The semantic knowledge baked into these embeddings by the Sentence-BERT pre-training is preserved. This makes the item representations content-aware from day one — even movies with no interactions will have a meaningful embedding based on their plot and title.

**b) Collaborative Filtering Item Embeddings (`nn.Embedding`)**

A trainable lookup table of size `(num_movies, latent_dim=64)`. These are the learnable collaborative filtering signals — they capture patterns like "users who liked Movie A also tended to like Movie B." They are initialized randomly and updated entirely through the training process.

**c) Shared Projection Network (`nn.Sequential`)**

Every item passes through this network regardless of whether it's a "user history item" or a "candidate item." It takes the concatenation of CF embedding (64-dim) and text embedding (384-dim), totaling 448 dimensions, and projects it to the shared 64-dimensional latent space:

```
448 → Linear → 256 → ReLU → Dropout(0.2) → Linear → 128 → ReLU → Linear → 64
```

The Dropout layer (rate 0.2) is applied during training to prevent overfitting. All items, whether used to represent the user or as candidates to score, go through this same projection — this is what allows the dot product comparison to be meaningful.

### 6.3 Forward Pass and Training Objective

During training, each example consists of a user history (`user_history_ids`), one positive item (`pos_item_ids`), and one negative item (`neg_item_ids`) sampled randomly from items the user has not interacted with.

1. **Get item representations:** Each history item, positive item, and negative item is passed through `get_item_rep()` which concatenates CF + text embeddings and applies the projection network.
2. **Form user representation:** `hist_reps.mean(dim=1)` — the user is the mean of their liked items' projected representations.
3. **Score positive and negative:** Dot product `(u_rep * pos_rep).sum(dim=1)` and `(u_rep * neg_rep).sum(dim=1)`.
4. **Loss:** Bayesian Personalized Ranking (BPR) loss — the model is trained to make `pos_score > neg_score` by a margin. This is a pairwise ranking objective, more appropriate than cross-entropy for implicit feedback recommendation.

### 6.4 Inference: Cached Item Representations

At inference time, the representations of all items in the catalog are computed **once** and cached:

```python
all_movie_idxs = torch.arange(cfg['num_movies']).to(_device)
_all_item_representations = _model.get_item_rep(all_movie_idxs)  # Shape: (10321, 64)
```

Recommendation then reduces to a single matrix multiplication: `user_vector @ _all_item_representations.T`, yielding a score for every movie in one GPU/CPU operation. This is what makes the Two-Tower retrieval extremely fast, regardless of catalog size.

---

## 7. Model Architecture: SASRec (Self-Attentive Sequential Recommender)

**Class:** `SASRec`

### 7.1 Design Philosophy

SASRec ([Kang & McAuley, 2018](https://arxiv.org/abs/1808.09781)) applies the Transformer architecture to sequential recommendation. Unlike the Two-Tower model which treats a user's history as an unordered bag, SASRec explicitly models the **temporal order** of interactions and learns which past items are most relevant for predicting the next one via self-attention.

This captures patterns like:
- After watching a comedy, a user often watches another comedy (recency bias)
- After the first film in a franchise, a user often watches the sequel next
- A user's interest shifts from action to drama over time

### 7.2 Components

**a) Item Embedding Layer (`nn.Embedding`)**

Size `(num_items + 1, embedding_dim)` with `padding_idx=0`. The +1 accommodates the 1-indexed item mapping (index 0 is always padding). `embedding_dim` is a hyperparameter (typically 64 or 128).

**b) Positional Embedding Layer (`nn.Embedding`)**

Size `(max_seq_length, embedding_dim)`. Unlike the original Transformer which uses fixed sinusoidal positional encodings, SASRec learns positional embeddings from data. Each position in the sequence (0 to 19) gets its own learnable vector. This is added to the item embedding before the Transformer encoder.

**c) Transformer Encoder**

A stack of `num_layers=2` standard Transformer encoder layers, each with:

| Parameter | Value |
|-----------|-------|
| `d_model` | `embedding_dim` |
| `nhead` | 2 |
| `dim_feedforward` | `hidden_dim` |
| `dropout` | 0.2 |
| `batch_first` | True |

The `batch_first=True` setting means inputs are shaped `(batch, sequence, features)` rather than `(sequence, batch, features)`.

**d) Output Linear Layer (`nn.Linear`)**

Projects the Transformer output from `embedding_dim` to `num_items + 1` — one logit per possible next item. This is used to compute softmax probabilities or directly for argmax-based top-k retrieval.

### 7.3 Causal Masking

A **causal (look-ahead) mask** is applied in the Transformer encoder:

```python
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
```

This is an upper-triangular mask that prevents position `i` from attending to any position `j > i`. This is critical because it enforces the **autoregressive property** — when predicting what comes after item at position 3, the model cannot look at items at positions 4, 5, ... 19. This makes training more honest and prevents the model from "cheating" by peeking at future interactions.

### 7.4 Output Head and Prediction

Only the **last position's output** is used for prediction:

```python
last_item_representation = x[:, -1, :]
logits = self.out(last_item_representation)
```

This is by design: given a padded sequence `[0, 0, ..., item_a, item_b, item_c]`, position `-1` contains the representation that has attended to (and aggregated information from) all preceding real items. Its projection through the output layer gives a score for every possible next item.

During inference, `torch.topk` is applied to these logits to get the top-k predicted next items.

---

## 8. The Two-Stage Recommendation Pipeline (Inference)

**Function:** `recommend(rated_movies, top_n=20, stage1_candidates=100)`

This is the central inference function. It orchestrates both models and the blending logic.

### 8.1 Input Contract

The function expects exactly **15 movie ratings** as a list of dicts:

```python
[
    {"movie_id": 318, "rating": 5.0},
    {"movie_id": 296, "rating": 4.0},
    ...  # 15 total
]
```

An assertion enforces this: `assert len(rated_movies) == 15`. The 15-rating requirement ensures there is enough signal for both models to work from, and provides a consistent user experience.

### 8.2 Stage 1: Candidate Generation (Two-Tower)

**Goal:** Reduce ~10,321 movies down to 100 high-quality candidates.

1. **Identify liked items:** All input movies with rating ≥ 4.0 are selected. If no movies meet this threshold (edge case), all 15 are used as a fallback.

2. **Map to Stage 1 indices:** Each liked MovieLens ID is mapped through `s1_mid2idx` to obtain the Two-Tower model's internal indices.

3. **Compute dynamic user embedding:**
   ```python
   liked_item_reps = _model.get_item_rep(user_history_tensor)
   dynamic_user_embedding = liked_item_reps.mean(dim=0, keepdim=True)  # Shape: (1, 64)
   ```

4. **Score all items:**
   ```python
   raw_scores = torch.matmul(dynamic_user_embedding, _all_item_representations.T).squeeze()
   ```
   This produces a 10,321-dimensional score vector in a single matrix multiply.

5. **Select top candidates:** `torch.topk` retrieves the top `100 + len(seen_mids)` indices. Already-rated movies are filtered out during iteration, yielding exactly 100 fresh candidates.

### 8.3 Stage 2: Sequential Intent Scoring (SASRec)

**Goal:** Identify the 5 movies that best match the user's *current sequential intent*.

1. **Extract the relevant sequence:** The system extracts only the user's **5-star movies** (rating == 5.0) from their 15 ratings, and takes the **last 5 of these** in the order they appear in the input. This is a deliberate design choice — 5-star movies represent the user's peak enthusiasm, and their relative order signals the current taste trajectory.

2. **Map to Stage 2 indices:** Each 5-star movie ID is converted to SASRec's 1-indexed item space via `s2_mid2idx`. Items not found in the mapping (outside SASRec's vocabulary) are silently skipped.

3. **Build padded input sequence:**
   ```python
   padded_seq = ([0] * (MAX_SEQ_LENGTH - len(seq_indices))) + seq_indices
   padded_seq = padded_seq[-MAX_SEQ_LENGTH:]  # Truncate if somehow too long
   ```
   This is left-padded with zeros to `MAX_SEQ_LENGTH = 20`.

4. **Run SASRec forward pass:**
   ```python
   logits = _sasrec_model(input_tensor)  # Shape: (1, num_items + 1)
   sasrec_scores = logits[0]
   ```

5. **Retrieve top-5 valid predictions:** The top 50 predicted indices are examined. The padding token (index 0) is skipped. Each index is mapped back to a MovieLens ID via `s2_idx2mid`. Already-rated movies are skipped. The first 5 valid new movies become the SASRec recommendations.

**Important implementation note:** The `s2_mid2idx` mapping is already 1-indexed (built with `{i: j+1 for ...}`), so no manual `+1` or `-1` offsets are needed when mapping between index space and movie ID space. This is explicitly commented in the code to prevent future bugs.

### 8.4 Blending: Sequel Boost → SASRec Top 5 → Two-Tower Fill

The blending logic combines the two model outputs with a heuristic layer:

**A) Franchise/Sequel Boost**

The system identifies the user's very last 5-star movie and extracts its "base franchise name" using the `get_base_franchise()` heuristic (detailed in Section 10.3). If a franchise name of 4+ characters is found, the system scans through ALL candidates (from both SASRec and Two-Tower) looking for any movie whose title contains that base franchise string. Matching movies are placed at the very top of the final list.

This is a lightweight but high-precision signal — if someone's most recent enthusiasm was for "The Dark Knight," the system should immediately surface "Batman Begins" or "The Dark Knight Rises" if the model hasn't already.

**B) SASRec Top-5 Insertion**

After the sequel boost, SASRec's top-5 results are inserted (if not already present from the sequel boost). These represent the most likely "next items" based on sequential patterns, capturing the user's immediate contextual intent.

**C) Two-Tower Fill**

The remaining positions (up to `top_n = 20`) are filled with Two-Tower candidates in ranked order, skipping any already present in the list.

**Final blend order:**
```
[sequel_matches] + [sasrec_top5] + [two_tower_fill_to_20]
```

### 8.5 Output Formatting

Each result is formatted as a dict:

```python
{
    "movie_id": int,
    "title": str,
    "genres": str,
    "year": int | None,
    "predicted_rating": float  # Descending dummy score for frontend rank ordering
}
```

The `predicted_rating` field is a simple descending rank score (`top_n - i`), not an actual predicted star rating. It exists solely to give the frontend a consistent field for sorting.

---

## 9. Model Loading: Lazy Singleton Pattern

Both models are loaded using a **lazy singleton pattern** — they are loaded at most once per server process and cached in module-level global variables.

```python
_model = None                    # Two-Tower model instance
_all_item_representations = None  # Cached item vectors (10321, 64)
_sasrec_model = None             # SASRec model instance
s2_mid2idx = None                # SASRec movie→index mapping
s2_idx2mid = None                # SASRec index→movie mapping
```

The loading functions `_load_stage1()` and `_load_stage2()` each begin with an early return guard:

```python
if _model is not None: return
```

The master `_load()` function calls both loaders in sequence and is invoked at the beginning of `recommend()`. This design means:
- The first request to the recommendation endpoint incurs the full model loading cost (loading weights from disk, moving to device, computing item cache)
- All subsequent requests are fast — only inference computation is needed
- Memory is not wasted loading models that are never used

The models are loaded from bundle files (`.pkl` files containing the model state dict, config, and metadata) specified via Django `settings` paths (`settings.INFERENCE_BUNDLE_PATH_25M` for Stage 1, `settings.INFERENCE_SASREC_BUNDLE_PATH` for Stage 2).

---

## 10. Supporting Utilities

### 10.1 Popular Movies

```python
def get_popular_movies(n=50)
```

Returns the top-N movies by `rating_count` from the Two-Tower model's metadata bundle. Used as a fallback or for homepage display before the user has rated anything. Only triggers Stage 1 loading (no need for SASRec here).

### 10.2 Search

```python
def search_movies(q, limit=20)
```

A fast in-memory text search over movie titles. Uses two-pass ranking:
1. **Prefix matches** (title starts with query string) — shown first
2. **Contains matches** (title contains query string anywhere) — shown second

Results are deduplicated and limited. This is an intentionally simple, non-ML approach — for title search, string matching is more precise than embedding similarity (which might surface semantically related but differently-titled movies).

### 10.3 Franchise Detection Heuristic

```python
def get_base_franchise(title) -> str
```

A rule-based heuristic to extract a canonical franchise name from a movie title. Applied in sequence:

1. Strip year annotations like `(1995)` via regex
2. Split on colons or dashes (e.g., `"Avengers: Age of Ultron"` → `"Avengers"`)
3. Strip trailing Arabic numerals (e.g., `"Toy Story 3"` → `"Toy Story"`)
4. Strip trailing Roman numerals (e.g., `"Rocky IV"` → `"Rocky"`)
5. Lowercase the result

A minimum franchise name length of **4 characters** is enforced before the boost is applied. This prevents short words like "The" or "Up" from spuriously matching unrelated movies.

Example transformations:

| Input Title | Output Franchise |
|-------------|-----------------|
| `Toy Story 3 (2010)` | `toy story` |
| `The Dark Knight (2008)` | `the dark knight` |
| `Avengers: Infinity War` | `avengers` |
| `Rocky IV` | `rocky` |
| `Star Wars: Episode IV - A New Hope` | `star wars` |

---

## 11. Key Design Decisions and Trade-offs

### Why Two Models Instead of One?

A single model cannot simultaneously optimize for both long-term preference consistency and short-term contextual intent. The Two-Tower model excels at the former because it aggregates over the entire history. SASRec excels at the latter because its attention mechanism can weight recent items more heavily and detect sequential patterns. Combining both captures what neither can alone.

### Why 15 Ratings as Input?

15 ratings provide:
- Enough items for the Two-Tower to build a stable mean user embedding (more items → more stable mean)
- Enough 5-star ratings in practice to give SASRec a meaningful sequence (at minimum, the last 1–5 five-star items)
- A concrete, bounded onboarding experience for the user

### Why Use Only 5-Star Movies for SASRec?

SASRec is a next-item predictor. If fed all 15 ratings regardless of score, its sequence would be polluted with movies the user actively disliked (1–3 stars). The model would then predict items *similar to bad experiences*, which is counterproductive. Using only 5-star movies keeps the sequence clean and focused on the user's strongest preferences.

### Why `all-MiniLM-L6-v2` for Text Embeddings?

This model offers an excellent efficiency/quality trade-off:
- Only 22M parameters (vs. 110M+ for BERT-base)
- 384-dim output (vs. 768-dim for larger models) — half the storage and compute
- Trained specifically for semantic similarity tasks via contrastive learning
- Fast enough to embed 10K+ movies in a few minutes on CPU
- Embeddings are frozen during Two-Tower training, so their quality directly determines the content-aware component of the recommendations

### Why Random Train/Val Split for Two-Tower but Leave-One-Out for SASRec?

The Two-Tower model is trained on individual (user, positive_item) pairs — the ordering of interactions doesn't matter. Random splitting is appropriate.

SASRec is trained to predict the *next item in a temporal sequence*. Using leave-one-out ensures the validation and test examples simulate the real inference scenario: given everything a user has done up to some point, predict what they'll do next. A random split would allow training examples to come *after* validation examples in time, creating future leakage.

### Trade-off: No Real-Time Feedback Loop

The current system generates recommendations from a single 15-rating cold-start session. There is no mechanism to update the models online as users provide more feedback. This is a common limitation of offline-trained systems and would be addressed in a production system with online learning or periodic retraining.

---

## 12. Complete Data and Model Flow Diagram (Textual)

```
RAW DATA
─────────────────────────────────────────────────────────────────────
  ml-25m/ratings.csv          ml-25m/movies.csv    ml-25m/links.csv
        │                           │                     │
        └───────────────────────────┴─────────────────────┘
                                    │
                        ┌───────────▼────────────┐
                        │  Two-Tower Data Prep   │
                        │  - Filter: ≥100 movie  │
                        │    ratings, ≥50 user   │
                        │    ratings             │
                        │  - Keep rating ≥ 4.0   │
                        │  - 0-index remap       │
                        │  - Join TMDB text      │
                        │  - Encode w/ MiniLM    │
                        └───────────┬────────────┘
                                    │
                   ┌────────────────┴──────────────────┐
                   │                                   │
        train_interactions.csv              movie_embeddings.pkl
        val_interactions.csv                movie_metadata.csv
        user2idx.pkl / movie2idx.pkl


  ml-25m/ratings.csv (all 25M)
        │
        ┌───────────▼────────────┐
        │  Sequential Data Prep  │
        │  - Keep ≥ 5 ratings    │
        │  - Sort chronological  │
        │  - 1-index remap       │
        │  - Sliding window      │
        │  - Left-pad to len 20  │
        │  - Leave-one-out split │
        └───────────┬────────────┘
                    │
       ┌────────────┴────────────┐
       │                        │
  train_data.pt              val_data.pt
  test_data.pt               mappings.pkl

MODEL TRAINING
─────────────────────────────────────────────────────────────────────

  train_interactions.csv  +  movie_embeddings.pkl
           │
           ▼
  HistoryBasedTwoTower (PyTorch)
  - Frozen MiniLM text embeddings (384-dim)
  - Trainable CF item embeddings (64-dim)
  - Projection network (448→256→128→64)
  - BPR pairwise loss
           │
           ▼
  inference_bundle_25m.pkl
  (model_state_dict + text_emb_matrix + movie_id_to_idx + movies_meta)

  train_data.pt (24.5M sequence samples)
           │
           ▼
  SASRec (PyTorch)
  - Item embeddings (59047+1, embed_dim)
  - Positional embeddings (20, embed_dim)
  - Transformer encoder (2 layers, 2 heads)
  - Linear output head (embed_dim → 59047+1)
  - Cross-entropy loss on next-item prediction
           │
           ▼
  sasrec_bundle.pkl
  (model_state_dict + hyperparameters)

INFERENCE
─────────────────────────────────────────────────────────────────────

  User Input: 15 rated movies
           │
           ├────────────────────────────────────────────────────────────────
           │  STAGE 1: Two-Tower                                           │
           │  1. liked_idxs ← ratings ≥ 4.0 → s1_mid2idx               │
           │  2. u_vec = mean(get_item_rep(liked_idxs))    (1, 64)       │
           │  3. scores = u_vec @ _all_item_reps.T         (10321,)      │
           │  4. top_k(scores, 100+seen) → filter seen → 100 candidates  │
           ├────────────────────────────────────────────────────────────────
           │  STAGE 2: SASRec                                             │
           │  1. Extract last 5 five-star movies                         │
           │  2. Map → s2_mid2idx → pad left to 20                      │
           │  3. logits = SASRec(padded_seq)              (59047+1,)     │
           │  4. top_k(logits, 50) → filter seen → 5 SASRec candidates  │
           ├────────────────────────────────────────────────────────────────
           │  BLENDING                                                    │
           │  1. Franchise boost from last 5-star movie                  │
           │  2. Insert SASRec top-5                                     │
           │  3. Fill to 20 with Two-Tower candidates                    │
           └────────────────────────────────────────────────────────────────
                                    │
                        ┌───────────▼─────────────┐
                        │  Final Output (top 20)  │
                        │  [{movie_id, title,     │
                        │   genres, year,         │
                        │   predicted_rating}]    │
                        └─────────────────────────┘
```

---

## 13. Hyperparameters and Configuration Summary

### Two-Tower Model

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `latent_dim` | 64 | Shared latent space dimensionality |
| `text_dim` | 384 | Fixed by `all-MiniLM-L6-v2` |
| `projection_layers` | 448→256→128→64 | With ReLU and Dropout(0.2) |
| `dropout` | 0.2 | Applied in projection network |
| `negative_sampling` | 1 negative per positive | BPR pairwise loss |
| `min_movie_ratings` | 100 | Popularity filter |
| `min_user_ratings` | 50 | Activity filter |
| `positive_threshold` | ≥ 4.0 stars | Implicit like signal |
| `text_encoder` | `all-MiniLM-L6-v2` | Frozen, 384-dim output |
| `train/val split` | 80/20 random | `random_state=42` |

### SASRec Model

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| `MAX_SEQ_LENGTH` | 20 | Maximum context window |
| `MIN_INTERACTIONS` | 5 | Minimum user interactions |
| `num_items` | 59,047 | Full catalog (all ratings) |
| `embedding_dim` | From bundle (`EMBEDDING_DIM`) | Typically 64 or 128 |
| `hidden_dim` | From bundle (`HIDDEN_DIM`) | Transformer FFN dim |
| `num_heads` | 2 | Multi-head self-attention |
| `num_layers` | 2 | Transformer encoder depth |
| `dropout` | 0.2 | Applied in Transformer layers |
| `padding_idx` | 0 | Reserved for left-padding |
| `index_base` | 1-indexed | `item2idx` starts at 1 |
| `split_strategy` | Leave-one-out per user | No future leakage |

### Inference Pipeline

| Parameter | Value |
|-----------|-------|
| `rated_movies` required | Exactly 15 |
| `stage1_candidates` | 100 |
| `sasrec_top_n` | 5 |
| `final_top_n` | 20 |
| `sasrec_input_sequence` | Last 5 five-star movies |
| `franchise_min_length` | 4 characters |
| `device` | Auto: CUDA if available, else CPU |

---

*Report prepared from source notebooks: `Data_Prep_2Tower.ipynb`, `Data_Prep_Sequential.ipynb`, `Model_Training_2Tower.ipynb`, `Model_Training_Sequential.ipynb`, and the production inference module.*
