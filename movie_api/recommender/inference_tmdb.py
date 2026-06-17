import pickle, numpy as np, pandas as pd
from django.conf import settings

_bundle = None

def _load():
    global _bundle
    if _bundle is not None:
        return
    with open(settings.INFERENCE_BUNDLE_PATH_TMDB, 'rb') as f:
        _bundle = pickle.load(f)

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

def find_similar(query, top_n=20):
    _load()
    meta  = _bundle['metadata']
    embs  = _bundle['embeddings']
    q_low = query.lower().strip()
    exact    = meta[meta['title'].str.lower() == q_low]
    starts   = meta[meta['title'].str.lower().str.startswith(q_low)]
    contains = meta[meta['title'].str.lower().str.contains(q_low, regex=False, na=False)]
    cands    = pd.concat([exact, starts, contains]).drop_duplicates('movie_id')
    if cands.empty:
        return {'error': f'No movie found matching "{query}"'}
    qrow = cands.iloc[0]
    qidx = int(qrow['row_idx'])
    qvec = embs[[qidx]]
    sims = (embs @ qvec.T).squeeze()
    sims[qidx] = -1.0
    top_idxs = sims.argsort()[::-1][:top_n]
    results = []
    for idx in top_idxs:
        r = meta.iloc[idx]
        results.append({'movie_id': int(r.movie_id), 'title': str(r.title),
                        'genres': str(r.genres),
                        'year': int(r.year) if r.year > 0 else None,
                        'similarity': round(float(sims[idx]), 4)})
    return {'query': {'movie_id': int(qrow.movie_id), 'title': str(qrow.title),
                        'genres': str(qrow.genres)},
            'similar': results}