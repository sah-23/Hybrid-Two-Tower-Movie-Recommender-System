# Movie Recommender System — Software Architecture Report

## Overview

The application is a full-stack web system that serves three distinct machine learning models through a unified user interface. It consists of a Next.js frontend, a Django REST API backend, and PyTorch models loaded from serialised inference bundles. The system is designed to run entirely locally with no external database — all data lives in memory after the server starts.

---

## Frontend

**Framework:** Next.js 14 (App Router)  
**Language:** TypeScript  
**Styling:** Tailwind CSS  
**HTTP client:** Axios

The frontend is organised under the `frontend/app/` directory using Next.js's file-system routing. Each route is a React Server Component shell with a `'use client'` page component inside that handles all user interaction.

### Pages and their roles

`/rate` — the primary rating page. The user fills in demographics (gender, age group, occupation), searches for movies by title using a debounced live search, adds 15 to a rated list with star ratings, and submits. On submission, the 15 rated movies and demographics are sent to `/api/recommend/`. The response array is stored in `sessionStorage` and the user is navigated to `/recommendations`.

`/rate-new` — identical layout to `/rate` but with no demographics panel. Searches against the 25M model's movie vocabulary via `/api/v2/movies/` and submits to `/api/v2/recommend/`. Results are stored under the same `sessionStorage` key so the same recommendations page handles both models.

`/recommendations` — reads the recommendations array from `sessionStorage` and renders a ranked list. The score bar normalises within the result set rather than against the 1–5 scale, so differences between closely ranked movies are visually meaningful. Each item shows title, year, genres, and a 6-decimal predicted score.

`/similar` — TMDB-based content similarity search. A search input with a live autocomplete dropdown calls `/api/tmdb/search/`. Selecting a suggestion or pressing Enter calls `/api/tmdb/similar/` and renders ranked results with cosine similarity percentages.

### API layer

All backend calls are centralised in `frontend/lib/api.ts`. An Axios instance is created with `baseURL: http://localhost:8000/api`. Each model has its own set of typed functions:

```
searchMovies()          → GET /api/movies/?q=
getRecommendations()    → POST /api/recommend/
searchMovies25m()       → GET /api/v2/movies/?q=
getRecommendations25m() → POST /api/v2/recommend/
searchMoviesTMDB()      → GET /api/tmdb/search/?q=
getSimilarTMDB()        → GET /api/tmdb/similar/?q=
```

TypeScript interfaces (`Movie`, `RatedMovie`, `Recommendation`, `Demographics`, `SimilarMoviesResult`) enforce the shape of every request and response at compile time.

---

## Backend

**Framework:** Django 4.x with Django REST Framework  
**Language:** Python 3.12  
**Serving:** Django development server (`manage.py runserver 8000`)

The backend project is structured as a single Django app called `recommender` inside `movie_api/`. There is no database — Django's default SQLite is created by migrations for internal session and auth tables only, and is never read or written by the recommender logic.

### URL routing

```
movie_api/config/urls.py
  └── /api/  →  recommender/urls.py
                  ├── movies/          →  movies_list
                  ├── recommend/       →  recommend
                  ├── similar/         →  similar_movies
                  ├── v2/movies/       →  movies_list_25m
                  ├── v2/recommend/    →  recommend_25m
                  ├── tmdb/search/     →  tmdb_search
                  └── tmdb/similar/    →  tmdb_similar
```

### Views

Each view is a plain function decorated with `@api_view`. Views do no ML work directly — they validate the request, call the appropriate inference module, and return a `Response`. The only logic in views is field-name normalisation (the `recommend` view accepts both `rated_movies` and `ratings` as field names for compatibility) and HTTP status mapping for error cases.

---

## Model Loading and Inference

### The inference bundle pattern

Each model is packaged into a single `.pkl` file (a Python pickle) that contains everything needed to serve predictions: model weights, architecture config, ID mappings, feature matrices, and movie metadata. No notebooks, no raw data files, and no imports from the training codebase are needed at serve time.

| Bundle | Path | Size |
|---|---|---|
| ML-1M model | `data/inference/inference_bundle.pkl` | ~45 MB |
| ML-25M model | `data/inference_25m/inference_bundle_25m.pkl` | ~415 MB |
| TMDB similarity | `data/inference_tmdb/inference_bundle_tmdb.pkl` | ~9 MB |

Bundle paths are declared in `movie_api/config/settings.py`:

```python
INFERENCE_BUNDLE_PATH      = BASE_DIR.parent / 'data' / 'inference' / 'inference_bundle.pkl'
INFERENCE_BUNDLE_PATH_25M  = BASE_DIR.parent / 'data' / 'inference_25m' / 'inference_bundle_25m.pkl'
INFERENCE_BUNDLE_PATH_TMDB = BASE_DIR.parent / 'data' / 'inference_tmdb' / 'inference_bundle_tmdb.pkl'
```

### Singleton loading

Each inference module (`inference.py`, `inference_25m.py`, `inference_tmdb.py`) uses a module-level singleton pattern. On the first request that hits any function in the module, `_load()` is called. It opens the pickle, reconstructs the PyTorch model from the stored state dict, moves it to the appropriate device, and stores references to the feature tensors in module-level globals. All subsequent requests reuse the already-loaded objects. Django's process model (single process, multiple threads) means the model is loaded once per server start and stays in RAM.

```python
_model  = None

def _load():
    global _model, _bundle, _genome_t, _rating_means_25m
    if _model is not None:
        return
    with open(settings.INFERENCE_BUNDLE_PATH_25M, 'rb') as f:
        _bundle = pickle.load(f)
    # reconstruct model, load weights, move to device
    ...
```

### Inference flow for a recommendation request

When a user submits 15 rated movies:

1. The view receives the POST body and extracts `rated_movies` (list of 15 dicts with `movie_id`, `rating`, `order`).
2. The inference module's `recommend()` function is called.
3. Rated movies are sorted by `order` (preserving the sequence the user rated them in, which is the sequential tower's input signal).
4. Movie IDs are mapped to contiguous indices using `movie_id_to_idx`.
5. Context tensors `ctx_m`, `ctx_r`, `ctx_p` are built as `[1, 15]` long tensors.
6. The rating encoder runs once and produces `collab_rep` (`[1, D]`), which is reused by both the NCF tower and (for the 1M model) the content tower.
7. The sequential tower runs once and produces `seq_score_base` (`[1]`), which is expanded to match the candidate batch size.
8. All candidate movies (excluding the 15 seen ones) are scored in batches of 512. For each batch: genome features are indexed from the pre-loaded tensor, content/NCF/genome scores are computed in parallel, the output head fuses the three scores.
9. For the 25M model, scores are demeaned predictions (deviation from item mean). Movies with fewer than 200 ratings are filtered out before ranking to avoid obscure films surfacing due to sparse signal.
10. Results are sorted by score descending, the top 20 are formatted with title/genres/year/predicted_rating and returned as JSON.

### CORS

`django-cors-headers` is installed with `CORS_ALLOW_ALL_ORIGINS = True` for local development. The `CorsMiddleware` must be the first entry in `MIDDLEWARE`.

---

## Data Flow Diagram

```
User browser
  │
  │  search query (keypress → 250ms debounce)
  ▼
Next.js (port 3000)
  │
  │  GET /api/movies/?q=spider
  ▼
Django REST API (port 8000)
  │
  │  inference module .search_movies("spider")
  │  → filters in-memory movies_meta DataFrame
  │  → returns [{movie_id, title, year, genres}, ...]
  ▼
Next.js renders dropdown

  │  user selects 15 movies, adjusts stars, submits
  ▼
  │  POST /api/recommend/ {demographics, ratings}
  ▼
Django REST API
  │
  │  inference module .recommend(demographics, ratings)
  │  → load bundle (cached after first call)
  │  → encode context → run three towers → batch score all candidates
  │  → rank → return top 20
  ▼
Next.js stores results in sessionStorage → navigates to /recommendations
  │
  ▼
/recommendations page reads sessionStorage → renders ranked list
```

---

## Project Directory Structure

```
project_root/
├── data/
│   ├── ml-1m/                    raw MovieLens 1M data
│   ├── ml-25m/                   raw MovieLens 25M data
│   ├── tmdb-5000/                raw TMDB 5000 data
│   ├── working/                  Data Prep artifacts (1M model)
│   ├── working_25m/              Data Prep artifacts (25M model)
│   ├── working_tmdb/             Data Prep artifacts (TMDB similarity)
│   ├── inference/
│   │   └── inference_bundle.pkl
│   ├── inference_25m/
│   │   └── inference_bundle_25m.pkl
│   └── inference_tmdb/
│       └── inference_bundle_tmdb.pkl
│
├── ML/
│   ├── Data_Prep.ipynb           1M model data preparation
│   ├── Model_Training.ipynb      1M model training
│   ├── Data_Prep_25M.ipynb       25M model data preparation
│   ├── Model_Training_25M.ipynb  25M model training
│   ├── Data_Prep_TMDB.ipynb      TMDB similarity data preparation
│   └── Model_Training_TMDB.ipynb TMDB similarity embedding + bundle
│
├── movie_api/                    Django backend
│   ├── config/
│   │   ├── settings.py
│   │   └── urls.py
│   └── recommender/
│       ├── views.py              REST endpoints
│       ├── urls.py               URL patterns
│       ├── inference.py          1M model inference singleton
│       ├── inference_25m.py      25M model inference singleton
│       ├── inference_tmdb.py     TMDB similarity inference singleton
│       └── ml/
│           └── predictor.py      1M model class definitions
│
└── frontend/                     Next.js frontend
    ├── app/
    │   ├── rate/page.tsx         1M model rating page
    │   ├── rate-new/page.tsx     25M model rating page
    │   ├── recommendations/page.tsx  results page
    │   └── similar/page.tsx      TMDB similarity page
    └── lib/
        └── api.ts                typed API functions
```

---

## Running the System

Two terminal windows are required:

**Terminal 1 — Django API:**
```bash
cd movie_api
python manage.py runserver 8000
```

**Terminal 2 — Next.js frontend:**
```bash
cd frontend
npm run dev
```

The application is then available at `http://localhost:3000`. The Django server loads all three inference bundles on first use — the 25M bundle (~415 MB) takes a few seconds on first request. All subsequent requests are fast because models stay in RAM.
