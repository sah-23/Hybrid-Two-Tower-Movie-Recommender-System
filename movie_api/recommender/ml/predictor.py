import pickle, torch
import torch.nn as nn
import numpy as np
from django.conf import settings
import pandas as pd

# ── Model classes (must match training exactly) ──────────────────────────────
class RatingEncoder(nn.Module):
    def __init__(self, num_movies, emb_dim, num_heads=4, shared_movie_emb=None):
        super().__init__()
        self.movie_emb   = shared_movie_emb or nn.Embedding(num_movies, emb_dim)
        self.rating_emb  = nn.Embedding(6, emb_dim, padding_idx=0)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.attention   = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=0.1)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(emb_dim * 2, emb_dim)
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
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.movie_mlp = nn.Sequential(
            nn.LayerNorm(movie_feat_dim), nn.Linear(movie_feat_dim, 256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, user_features_raw, movie_features, collab_rep):
        g = self.gender_emb(user_features_raw[:, 0].long())
        a = self.age_emb(user_features_raw[:, 1].long())
        o = self.occupation_emb(user_features_raw[:, 2].long())
        demo     = torch.cat([g, a, o], dim=1)
        user_rep = self.user_mlp(torch.cat([demo, collab_rep], dim=1))
        mov_rep  = self.movie_mlp(movie_features)
        return (user_rep * mov_rep).sum(dim=1)


class NCFTower(nn.Module):
    def __init__(self, num_movies, emb_dim, shared_movie_emb=None):
        super().__init__()
        self.rating_encoder   = RatingEncoder(num_movies, emb_dim, shared_movie_emb=shared_movie_emb)
        self.target_movie_emb = shared_movie_emb or nn.Embedding(num_movies, emb_dim)
        self.gmf_linear = nn.Linear(emb_dim, 32)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(), nn.Dropout(0.2)
        )
        self.fusion = nn.Linear(64, 1)

    def score_from_user_rep(self, user_rep, target_movie_id):
        m   = self.target_movie_emb(target_movie_id)
        gmf = self.gmf_linear(user_rep * m)
        mlp = self.mlp(torch.cat([user_rep, m], dim=1))
        return self.fusion(torch.cat([gmf, mlp], dim=1)).squeeze(1)


class SequentialTower(nn.Module):
    def __init__(self, num_movies, seq_len, emb_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.movie_emb   = nn.Embedding(num_movies, emb_dim, padding_idx=0)
        self.rating_emb  = nn.Embedding(6, emb_dim, padding_idx=0)
        self.pos_emb     = nn.Embedding(seq_len, emb_dim)
        self.emb_dropout = nn.Dropout(0.3)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads,
            dim_feedforward=emb_dim * 4, dropout=0.3, batch_first=True
        )
        self.transformer    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attention_pool = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True, dropout=0.2)
        self.out_dropout    = nn.Dropout(0.3)
        self.score_head     = nn.Linear(emb_dim, 1)

    def forward(self, movie_ids, ratings, positions):
        x = self.emb_dropout(
            self.movie_emb(movie_ids) + self.rating_emb(ratings) + self.pos_emb(positions)
        )
        mask = nn.Transformer.generate_square_subsequent_mask(
            movie_ids.shape[1], device=movie_ids.device, dtype=x.dtype
        )
        out    = self.transformer(x, mask=mask, src_key_padding_mask=None, is_causal=False)
        query  = out.mean(dim=1, keepdim=True)
        pooled, _ = self.attention_pool(query, out, out)
        return self.score_head(self.out_dropout(pooled.squeeze(1))).squeeze(1)


class HybridRecommender(nn.Module):
    def __init__(self, num_movies, movie_feat_dim, emb_dim=128, seq_len=15):
        super().__init__()
        self.movie_emb     = nn.Embedding(num_movies, emb_dim, padding_idx=0)
        self.content_tower = ContentTower(movie_feat_dim, emb_dim)
        self.ncf_tower     = NCFTower(num_movies, emb_dim, shared_movie_emb=self.movie_emb)
        self.seq_tower     = SequentialTower(num_movies, seq_len, emb_dim)
        self.fusion_weights = nn.Parameter(torch.tensor([0.4, 0.4, 0.2]))
        self.output_head    = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, uf, ctx_m, ctx_r, ctx_p, tgt_m, tmf):
        cr      = self.ncf_tower.rating_encoder(ctx_m, ctx_r)
        c_score = self.content_tower(uf, tmf, cr)
        n_score = self.ncf_tower.score_from_user_rep(cr, tgt_m)
        s_score = self.seq_tower(ctx_m, ctx_r, ctx_p)
        scores  = torch.stack([c_score, n_score, s_score], dim=1)
        out     = self.output_head(scores).squeeze(1)
        if not self.training:
            out = out.clamp(1.0, 5.0)
        return out


# ── Singleton loader ─────────────────────────────────────────────────────────
_predictor = None

GENDER_MAP = {'M': 0, 'F': 1, 'm': 0, 'f': 1}
AGE_MAP    = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class Predictor:
    def __init__(self):
        path = settings.INFERENCE_BUNDLE_PATH
        print(f"[Predictor] Loading bundle from {path} ...")
        with open(path, 'rb') as f:
            bundle = pickle.load(f)

        cfg = bundle['model_config']
        self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.movie_id_to_idx = bundle['movie_id_to_idx']
        self.idx_to_movie_id = bundle['idx_to_movie_id']
        self.movies_enriched = bundle['movies_enriched']
        self.seq_len         = cfg['seq_len']

        # Feature matrices
        self.movie_features = torch.tensor(
            bundle['movie_features_np'], dtype=torch.float32
        )

        # Rebuild and load model
        self.model = HybridRecommender(
            num_movies=cfg['num_movies'],
            movie_feat_dim=cfg['movie_feat_dim'],
            emb_dim=cfg['emb_dim'],
            seq_len=cfg['seq_len'],
        )
        self.model.load_state_dict(bundle['model_state_dict'])
        self.model.to(self.device).eval()
        print(f"[Predictor] Model ready on {self.device}")

    def _encode_demographics(self, gender, age, occupation):
        g = GENDER_MAP.get(str(gender), 0)
        a = AGE_MAP.get(int(age), 2)       # default to 25-group if unknown
        o = max(0, min(20, int(occupation)))
        return torch.tensor([[g, a, o]], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def recommend(self, demographics, ratings_input, top_n=20, exclude_seen=True):
        """
        demographics  : {'gender': 'M'/'F', 'age': int, 'occupation': int}
        ratings_input : [{'movie_id': int, 'rating': 1-5, 'order': 0-14}, ...]
        """
        rated    = sorted(ratings_input, key=lambda x: x['order'])
        seen_ids = set()
        ctx_m, ctx_r, ctx_p = [], [], []
        for i, entry in enumerate(rated):
            mid = int(entry['movie_id'])
            if mid not in self.movie_id_to_idx:
                raise ValueError(f"movie_id {mid} not in vocabulary")
            ctx_m.append(self.movie_id_to_idx[mid])
            ctx_r.append(max(1, min(5, int(entry['rating']))))
            ctx_p.append(i)
            seen_ids.add(mid)

        ctx_m_t = torch.tensor(ctx_m, dtype=torch.long).unsqueeze(0).to(self.device)
        ctx_r_t = torch.tensor(ctx_r, dtype=torch.long).unsqueeze(0).to(self.device)
        ctx_p_t = torch.tensor(ctx_p, dtype=torch.long).unsqueeze(0).to(self.device)
        uf      = self._encode_demographics(
            demographics['gender'], demographics['age'], demographics['occupation']
        )

        # Pre-compute user reps once
        collab_rep     = self.model.ncf_tower.rating_encoder(ctx_m_t, ctx_r_t)
        seq_score_base = self.model.seq_tower(ctx_m_t, ctx_r_t, ctx_p_t)

        candidate_ids = [m for m in self.movie_id_to_idx
                         if not (exclude_seen and m in seen_ids)]

        BATCH   = 512
        results = []
        for start in range(0, len(candidate_ids), BATCH):
            batch_mids = candidate_ids[start:start + BATCH]
            B          = len(batch_mids)
            bidxs      = torch.tensor(
                [self.movie_id_to_idx[m] for m in batch_mids],
                dtype=torch.long, device=self.device
            )
            tmf    = self.movie_features[bidxs.cpu()].to(self.device)
            uf_b   = uf.expand(B, -1)
            cr_b   = collab_rep.expand(B, -1)
            c_s    = self.model.content_tower(uf_b, tmf, cr_b)
            n_s    = self.model.ncf_tower.score_from_user_rep(cr_b, bidxs)
            s_s    = seq_score_base.expand(B)
            stacked = torch.stack([c_s, n_s, s_s], dim=1)
            final   = self.model.output_head(stacked).squeeze(1).clamp(1.0, 5.0)
            results.extend(zip(batch_mids, final.cpu().numpy()))

        results.sort(key=lambda x: x[1], reverse=True)
        movie_info = self.movies_enriched.set_index('MovieID')
        output = []
        for mid, score in results[:top_n]:
            row = movie_info.loc[mid] if mid in movie_info.index else None
            output.append({
                'movie_id'        : int(mid),
                'title'           : str(row['Title_Clean']) if row is not None else f'Movie {mid}',
                'genres'          : str(row['Genres_ML'])   if row is not None else '',
                'predicted_rating': round(float(score), 6),
            })
        return output

    def get_all_movies(self):
        """Return all movies for the rating UI."""
        rows = []
        for _, row in self.movies_enriched.iterrows():
            mid = int(row['MovieID'])
            if mid not in self.movie_id_to_idx:
                continue
            rows.append({
                'movie_id': mid,
                'title'   : str(row['Title_Clean']),
                'genres'  : str(row['Genres_ML']),
                'year'    : int(row['Year']) if str(row['Year']).isdigit() else None,
            })
        rows.sort(key=lambda x: x['title'])
        return rows
    
    @torch.no_grad()
    def similar_movies(self, query_title: str, top_n: int = 20):
        """
        Find movies with similar content (plot, themes, tone) to query_title.
        Uses cosine similarity over SBERT overview embeddings.
        Works for any movie in movies_enriched, including TMDB-only movies
        that were not in the ML-1M training set.
        """
        df = self.movies_enriched

        # ── Find the query movie ─────────────────────────────────────────────────
        title_lower   = query_title.lower().strip()
        df['_match']  = df['Title_Clean'].str.lower().str.strip()

        # Exact match first, then startswith, then contains
        exact    = df[df['_match'] == title_lower]
        starts   = df[df['_match'].str.startswith(title_lower)]
        contains = df[df['_match'].str.contains(title_lower, regex=False, na=False)]

        candidates = pd.concat([exact, starts, contains]).drop_duplicates('MovieID')
        df.drop(columns='_match', inplace=True)

        if candidates.empty:
            return {'error': f'No movie found matching "{query_title}"'}

        query_row = candidates.iloc[0]
        query_mid = int(query_row['MovieID'])

        # ── Get the query movie's feature vector ─────────────────────────────────
        # movie_features has shape [NUM_MOVIES, feat_dim] aligned to movie_id_to_idx.
        # For TMDB-only movies (not in idx), fall back to their stored feature vector
        # directly from movies_enriched if available, else return an error.
        if query_mid in self.movie_id_to_idx:
            query_idx = self.movie_id_to_idx[query_mid]
            query_vec = self.movie_features[query_idx].unsqueeze(0)   # [1, D]
        else:
            return {'error': f'"{query_row["Title_Clean"]}" is not in the model vocabulary'}

        # ── Cosine similarity against all vocab movies ────────────────────────────
        # Normalise both query and all movie vectors, then dot product = cosine sim
        query_norm  = torch.nn.functional.normalize(query_vec, dim=1)              # [1, D]
        movies_norm = torch.nn.functional.normalize(self.movie_features, dim=1)   # [N, D]
        sims        = (movies_norm @ query_norm.T).squeeze(1)                      # [N]

        # Exclude the query movie itself
        sims[query_idx] = -1.0

        top_k_vals, top_k_idxs = torch.topk(sims, k=min(top_n, len(sims)))

        movie_info = self.movies_enriched.set_index('MovieID')
        results    = []
        for idx, sim in zip(top_k_idxs.tolist(), top_k_vals.tolist()):
            mid = self.idx_to_movie_id[idx]
            row = movie_info.loc[mid] if mid in movie_info.index else None
            results.append({
                'movie_id'  : int(mid),
                'title'     : str(row['Title_Clean']) if row is not None else f'Movie {mid}',
                'genres'    : str(row['Genres_ML'])   if row is not None else '',
                'year'      : int(row['Year']) if row is not None and
                            str(row.get('Year', '')).lstrip('-').isdigit() else None,
                'similarity': round(float(sim), 4),
            })

        return {
            'query'  : {
                'movie_id': query_mid,
                'title'   : str(query_row['Title_Clean']),
                'genres'  : str(query_row['Genres_ML']),
            },
            'similar': results,
        }