# movie_api/recommender/inference.py
import pickle, torch, numpy as np
import torch.nn as nn
from django.conf import settings

# ── Model architecture (must match training exactly) ──────────────────────────

class RatingEncoder(nn.Module):
    def __init__(self, num_movies, num_ratings=6, emb_dim=64, num_heads=4, shared_movie_emb=None):
        super().__init__()
        self.movie_emb   = shared_movie_emb or nn.Embedding(num_movies, emb_dim)
        self.rating_emb  = nn.Embedding(num_ratings, emb_dim, padding_idx=0)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.attention   = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=0.1)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, movie_ids, ratings):
        B        = movie_ids.shape[0]
        combined = self.dropout(self.movie_emb(movie_ids) + self.rating_emb(ratings))
        cls      = self.cls_token.expand(B, -1, -1)
        combined = torch.cat([cls, combined], dim=1)
        attn_out, _ = self.attention(combined, combined, combined)
        combined = self.layer_norm1(attn_out + combined)
        combined = self.layer_norm2(self.ffn(combined) + combined)
        return combined[:, 0, :]


class ContentTower(nn.Module):
    def __init__(self, movie_feat_dim, hidden_dim):
        super().__init__()
        self.gender_emb     = nn.Embedding(3,  8)
        self.age_emb        = nn.Embedding(8,  16)
        self.occupation_emb = nn.Embedding(22, 32)
        self.user_mlp = nn.Sequential(
            nn.Linear(56 + hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.movie_mlp = nn.Sequential(
            nn.LayerNorm(movie_feat_dim), nn.Linear(movie_feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, user_features_raw, movie_features, collab_rep):
        g = self.gender_emb(user_features_raw[:, 0].long())
        a = self.age_emb(user_features_raw[:, 1].long())
        o = self.occupation_emb(user_features_raw[:, 2].long())
        user_input = torch.cat([torch.cat([g, a, o], dim=1), collab_rep], dim=1)
        return (self.user_mlp(user_input) * self.movie_mlp(movie_features)).sum(dim=1)


class NCFTower(nn.Module):
    def __init__(self, num_movies, emb_dim=64, shared_movie_emb=None):
        super().__init__()
        self.rating_encoder   = RatingEncoder(num_movies, emb_dim=emb_dim, shared_movie_emb=shared_movie_emb)
        self.target_movie_emb = shared_movie_emb or nn.Embedding(num_movies, emb_dim)
        self.gmf_linear = nn.Linear(emb_dim, 32)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.fusion = nn.Linear(64, 1)

    def score_from_user_rep(self, user_rep, target_movie_id):
        movie_rep = self.target_movie_emb(target_movie_id)
        return self.fusion(torch.cat([
            self.gmf_linear(user_rep * movie_rep),
            self.mlp(torch.cat([user_rep, movie_rep], dim=1))
        ], dim=1)).squeeze(1)

    def forward(self, ctx_movie_ids, ctx_ratings, target_movie_id):
        return self.score_from_user_rep(
            self.rating_encoder(ctx_movie_ids, ctx_ratings), target_movie_id)


class SequentialTower(nn.Module):
    def __init__(self, num_movies, seq_len=15, emb_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.movie_emb   = nn.Embedding(num_movies, emb_dim, padding_idx=0)
        self.rating_emb  = nn.Embedding(6, emb_dim, padding_idx=0)
        self.pos_emb     = nn.Embedding(seq_len, emb_dim)
        self.emb_dropout = nn.Dropout(0.3)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads,
            dim_feedforward=emb_dim * 4, dropout=0.3, batch_first=True,
        )
        self.transformer    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=0.2)
        self.out_dropout    = nn.Dropout(0.3)
        self.score_head     = nn.Linear(emb_dim, 1)

    def forward(self, movie_ids, ratings, positions):
        seq_input   = self.emb_dropout(
            self.movie_emb(movie_ids) + self.rating_emb(ratings) + self.pos_emb(positions)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            movie_ids.shape[1], device=movie_ids.device, dtype=seq_input.dtype
        )
        seq_out   = self.transformer(seq_input, mask=causal_mask,
                                     src_key_padding_mask=None, is_causal=False)
        query     = seq_out.mean(dim=1, keepdim=True)
        pooled, _ = self.attention_pool(query, seq_out, seq_out)
        return self.score_head(self.out_dropout(pooled.squeeze(1))).squeeze(1)


class HybridRecommender(nn.Module):
    def __init__(self, num_movies, movie_feat_dim, emb_dim=64):
        super().__init__()
        self.movie_emb     = nn.Embedding(num_movies, emb_dim, padding_idx=0)
        self.content_tower = ContentTower(movie_feat_dim, hidden_dim=emb_dim)
        self.ncf_tower     = NCFTower(num_movies, emb_dim=emb_dim, shared_movie_emb=self.movie_emb)
        self.seq_tower     = SequentialTower(num_movies, seq_len=15, emb_dim=emb_dim)
        self.fusion_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))
        self.output_head   = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, user_features, ctx_movie_ids, ctx_ratings, ctx_positions,
                target_movie_id, target_movie_features):
        collab = self.ncf_tower.rating_encoder(ctx_movie_ids, ctx_ratings)
        scores = torch.stack([
            self.content_tower(user_features, target_movie_features, collab),
            self.ncf_tower.score_from_user_rep(collab, target_movie_id),
            self.seq_tower(ctx_movie_ids, ctx_ratings, ctx_positions),
        ], dim=1)
        final = self.output_head(scores).squeeze(1)
        if not self.training:
            final = final.clamp(1.0, 5.0)
        return final, scores[:, 0], scores[:, 1], scores[:, 2]


# ── Singleton loader — model is loaded once when Django starts ────────────────

GENDER_MAP = {'M': 0, 'F': 1}
AGE_MAP    = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

_bundle = None
_model  = None
_device = None
_movie_features_tensor = None
_popular_movies_cache  = None


def _load():
    global _bundle, _model, _device, _movie_features_tensor
    if _model is not None:
        return

    with open(settings.INFERENCE_BUNDLE_PATH, 'rb') as f:
        _bundle = pickle.load(f)

    cfg    = _bundle['model_config']
    _device = torch.device('cpu')   # CPU for serving; switch to cuda if GPU available

    _model = HybridRecommender(
        num_movies=cfg['num_movies'],
        movie_feat_dim=cfg['movie_feat_dim'],
        emb_dim=cfg['emb_dim'],
    )
    _model.load_state_dict(_bundle['model_state_dict'])
    _model.eval()

    _movie_features_tensor = torch.tensor(
        _bundle['movie_features_np'], dtype=torch.float32
    )


def get_popular_movies(n=50):
    """Return n popular movies the UI will display for the user to rate."""
    global _popular_movies_cache
    _load()
    if _popular_movies_cache is not None:
        return _popular_movies_cache

    df = _bundle['movies_enriched']
    # Sort by vote_count descending to pick well-known movies
    if 'vote_count' in df.columns:
        df_sorted = df[df['vote_count'].notna()].sort_values('vote_count', ascending=False)
    else:
        df_sorted = df

    results = []
    for _, row in df_sorted.head(n).iterrows():
        mid = int(row['MovieID'])
        if mid not in _bundle['movie_id_to_idx']:
            continue
        results.append({
            'movie_id': mid,
            'title'   : str(row.get('Title_Clean', row.get('Title_Original', f'Movie {mid}'))),
            'genres'  : str(row.get('Genres_ML', '')),
            'year'    : int(row['Year']) if 'Year' in row and row['Year'] == row['Year'] else None,
        })
        
    seen = set()
    results = [r for r in results if r['movie_id'] not in seen and not seen.add(r['movie_id'])]
    _popular_movies_cache = results
    return results

def search_movies(q: str, limit: int = 20) -> list:
    """Search movies by title substring, starts-with ranked first."""
    _load()
    q_lower = q.lower().strip()
    df      = _bundle['movies_enriched']
    mid2idx = _bundle['movie_id_to_idx']

    results = []
    for _, row in df.iterrows():
        mid = int(row['MovieID'])
        if mid not in mid2idx:
            continue
        title = str(row.get('Title_Clean', row.get('Title_Original', '')))
        if q_lower in title.lower():
            results.append({
                'movie_id': mid,
                'title'   : title,
                'genres'  : str(row.get('Genres_ML', '')),
                'year'    : int(row['Year']) if str(row.get('Year', '')).isdigit() else None,
                '_starts' : title.lower().startswith(q_lower),
            })

    # starts-with first, then contains
    # Deduplicate by movie_id — keep first occurrence after sort
    seen = set()
    deduped = []
    results.sort(key=lambda x: (not x['_starts'], x['title']))
    for r in results:
        del r['_starts']
        if r['movie_id'] not in seen:
            seen.add(r['movie_id'])
            deduped.append(r)
    return deduped[:limit]



@torch.no_grad()
def recommend(demographics, rated_movies, top_n=20):
    """
    demographics : {'gender': 'M'|'F', 'age': int, 'occupation': int}
    rated_movies : [{'movie_id': int, 'rating': 1-5, 'order': 0-14}, ...] — exactly 15
    Returns      : [{'movie_id', 'title', 'genres', 'predicted_rating'}, ...]
    """
    _load()
    mid2idx = _bundle['movie_id_to_idx']
    movies  = _bundle['movies_enriched']

    rated_sorted = sorted(rated_movies, key=lambda x: x['order'])
    assert len(rated_sorted) == 15, "Exactly 15 ratings required"

    seen, ctx_m, ctx_r, ctx_p = set(), [], [], []
    for i, entry in enumerate(rated_sorted):
        mid = int(entry['movie_id'])
        ctx_m.append(mid2idx[mid])
        ctx_r.append(max(1, min(5, int(entry['rating']))))
        ctx_p.append(i)
        seen.add(mid)

    ctx_m_t = torch.tensor(ctx_m, dtype=torch.long).unsqueeze(0)
    ctx_r_t = torch.tensor(ctx_r, dtype=torch.long).unsqueeze(0)
    ctx_p_t = torch.tensor(ctx_p, dtype=torch.long).unsqueeze(0)

    g = GENDER_MAP.get(str(demographics['gender']).upper(), 0)
    a = AGE_MAP.get(int(demographics['age']), 0)
    o = max(0, min(20, int(demographics['occupation'])))
    uf = torch.tensor([[g, a, o]], dtype=torch.long)

    collab_rep     = _model.ncf_tower.rating_encoder(ctx_m_t, ctx_r_t)
    seq_score_base = _model.seq_tower(ctx_m_t, ctx_r_t, ctx_p_t)

    candidates = [m for m in mid2idx if m not in seen]
    BATCH, all_scores = 512, []

    for start in range(0, len(candidates), BATCH):
        batch = candidates[start : start + BATCH]
        B     = len(batch)
        bidxs = torch.tensor([mid2idx[m] for m in batch], dtype=torch.long)
        tmf   = _movie_features_tensor[bidxs]
        cr_b  = collab_rep.expand(B, -1)
        uf_b  = uf.expand(B, -1)

        content_s = _model.content_tower(uf_b, tmf, cr_b)
        ncf_s     = _model.ncf_tower.score_from_user_rep(cr_b, bidxs)
        seq_s     = seq_score_base.expand(B)

        stacked = torch.stack([content_s, ncf_s, seq_s], dim=1)
        final   = _model.output_head(stacked).squeeze(1).clamp(1.0, 5.0)
        all_scores.extend(zip(batch, final.numpy()))

    all_scores.sort(key=lambda x: x[1], reverse=True)
    movie_info = movies.set_index('MovieID')
    def parse_year_1m(row):
        try:
            return int(float(str(row.get('Year', ''))))
        except (ValueError, TypeError):
            return None
        
    results = []
    for mid, score in all_scores[:top_n]:
        row = movie_info.loc[mid] if mid in movie_info.index else None


        results.append({
            'movie_id'        : int(mid),
            'title'           : str(row['Title_Clean']) if row is not None else f'Movie {mid}',
            'genres'          : str(row['Genres_ML'])   if row is not None else '',
            'year'            : parse_year_1m(row)      if row is not None else None,
            'predicted_rating': round(float(score), 6),
        })
    return results