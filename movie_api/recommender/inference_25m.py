import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from django.conf import settings
import re




class HistoryBasedTwoTower(nn.Module):
    def __init__(self, num_users, num_movies, text_emb_matrix, latent_dim=64):
        super().__init__()
        # Text Embeddings (Frozen, from all-MiniLM)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb_matrix, freeze=True)
        text_dim = text_emb_matrix.shape[1] # 384
        
        # Collaborative Filtering Embeddings (Only Items!)
        self.item_emb = nn.Embedding(num_movies, latent_dim)
        
        # Shared Projection Network to map item features to the latent space
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + text_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def get_item_rep(self, item_ids):
        i_cf = self.item_emb(item_ids)
        i_text = self.text_embeddings(item_ids)
        i_concat = torch.cat([i_cf, i_text], dim=1)
        return self.proj(i_concat)

    def forward(self, user_history_ids, pos_item_ids, neg_item_ids):
        # In history-based tower, user rep is the mean of their history's item representations
        hist_reps = self.get_item_rep(user_history_ids)
        u_rep = hist_reps.mean(dim=1) 
        
        pos_rep = self.get_item_rep(pos_item_ids)
        neg_rep = self.get_item_rep(neg_item_ids)
        
        # Dot product for scores
        pos_scores = (u_rep * pos_rep).sum(dim=1)
        neg_scores = (u_rep * neg_rep).sum(dim=1)
        return pos_scores, neg_scores

# ─── STAGE 2 MODEL: SASRec (SEQUENTIAL) ─────────────────────────────────────────
# This model is good at predicting the very next item a user might interact with.

class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim, num_heads=2, num_layers=2, dropout=0.2, max_seq_length=20):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(embedding_dim, num_items + 1)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Create causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        item_embed = self.item_emb(x)
        pos_embed = self.pos_emb(positions)
        
        x = item_embed + pos_embed
        x = self.transformer_encoder(x, mask=causal_mask)
        
        # We only care about the output of the LAST item in the sequence
        last_item_representation = x[:, -1, :]
        logits = self.out(last_item_representation)
        return logits


# ── Singleton Variables ───────────────────────────────────────────────────────
_bundle    = None
_model     = None
_device    = None
_all_item_representations = None # Cache for fast Stage 1 inference

_sasrec_bundle = None
_sasrec_model = None
s2_mid2idx = None
s2_idx2mid = None

def _load_stage1():
    """Loads the Two-Tower model for candidate generation."""
    global _bundle, _model, _device, _all_item_representations
    if _model is not None: return

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(settings.INFERENCE_BUNDLE_PATH_25M, 'rb') as f:
        _bundle = torch.load(f, map_location=_device, weights_only=False)
    
    cfg = _bundle['model_config']
    text_emb_matrix = torch.tensor(_bundle['text_emb_matrix'], dtype=torch.float32)
    
    _model = HistoryBasedTwoTower(
        num_users=cfg['num_users'],
        num_movies=cfg['num_movies'],
        text_emb_matrix=text_emb_matrix,
        latent_dim=cfg.get('latent_dim', 64) 
    ).to(_device)
    _model.load_state_dict(_bundle['model_state_dict'])
    _model.eval()

    all_movie_idxs = torch.arange(cfg['num_movies']).to(_device)
    with torch.no_grad():
        _all_item_representations = _model.get_item_rep(all_movie_idxs)
    print(f'[Stage 1] Two-Tower model loaded successfully on {_device}')

import os # Make sure this is at the top of your file

# ... [Keep your STAGE 1 and STAGE 2 Model Classes as they are] ...

# Add the mapping variables to your Singleton Variables section
_bundle    = None
_model     = None
_device    = None
_all_item_representations = None

_sasrec_bundle = None
_sasrec_model = None
s2_mid2idx = None
s2_idx2mid = None

# ... [Keep _load_stage1 as it is] ...

def _load_stage2():
    """Loads the SASRec model and its mappings for re-ranking."""
    global _sasrec_bundle, _sasrec_model, _device, s2_mid2idx, s2_idx2mid
    if _sasrec_model is not None and s2_mid2idx is not None: return
    
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the mappings using Django's BASE_DIR to ensure the path is always correct
    mappings_path = os.path.join(settings.BASE_DIR.parent, 'data', 'inference_seq', 'mappings.pkl')
    print("Loading SASRec mappings from:", mappings_path)
    with open(mappings_path, 'rb') as f:
        sasrec_mappings = pickle.load(f)
        s2_mid2idx = sasrec_mappings['item2idx']
        s2_idx2mid = {idx: mid for mid, idx in s2_mid2idx.items()}

    # 2. Load the Model Bundle
    with open(settings.INFERENCE_SASREC_BUNDLE_PATH, 'rb') as f:
        _sasrec_bundle = torch.load(f, map_location=_device)
    
    cfg = _sasrec_bundle['hyperparameters']
    _sasrec_model = SASRec(
        num_items=_sasrec_bundle['hyperparameters']['NUM_ITEMS']- 1,
        embedding_dim=cfg['EMBEDDING_DIM'],
        hidden_dim=cfg['HIDDEN_DIM'],
        max_seq_length=cfg['MAX_SEQ_LENGTH']
    ).to(_device)
    
    try:
        _sasrec_model.load_state_dict(_sasrec_bundle['model_state_dict'])
    except Exception as e:
        print("\n!!! ERROR LOADING STATE DICT !!!")
        print(e)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        raise e
    
    _sasrec_model.eval()
    print(f'[Stage 2] SASRec model and mappings loaded successfully on {_device}')

    print("SASREC BUNDLE KEYS:", _sasrec_bundle.keys())

def _load():
    """Master loading function to initialize both models."""
    _load_stage1()
    _load_stage2()


# ── Existing Functionality (Search & Popular) ─────────────────────────────────
# ... (No changes to get_popular_movies or search_movies, keeping them as is)
def get_popular_movies(n=50):
    _load_stage1() # Only needs stage 1 model
    meta = _bundle['movies_meta']
    top  = meta.sort_values('rating_count', ascending=False).head(n)
    seen, out = set(), []
    for _, row in top.iterrows():
        mid = int(row['MovieID'])
        if mid in seen: continue
        seen.add(mid)
        out.append({
            'movie_id': mid,
            'Title_Clean'   : str(row['Title_Clean']),
            'genres'  : str(row.get('Genres', '')),
            'year'    : int(float(row['Year'])) if str(row.get('Year', '')).replace('.', '', 1).lstrip('-').isdigit() else None,
        })
    return out

def search_movies(q, limit=20):
    _load_stage1() # Only needs stage 1 model
    if _bundle is None: return []
    meta    = _bundle['movies_meta']
    q_lower = q.lower().strip()
    starts   = meta[meta['Title_Clean'].str.lower().str.startswith(q_lower, na=False)]
    contains = meta[
        ~meta['Title_Clean'].str.lower().str.startswith(q_lower, na=False) &
        meta['Title_Clean'].str.lower().str.contains(q_lower, regex=False, na=False)
    ]
    combined = pd.concat([starts, contains]).drop_duplicates('MovieID').head(limit)
    seen, out = set(), []
    for _, row in combined.iterrows():
        mid = int(row['MovieID'])
        if mid in seen: continue
        seen.add(mid)
        out.append({
            'movie_id': mid,
            'title'   : str(row['Title_Clean']),
            'genres'  : str(row.get('Genres', '')),
            'year'    : int(float(row['Year'])) if str(row.get('Year', '')).replace('.', '', 1).lstrip('-').isdigit() else None,
        })
    return out

def get_base_franchise(title):
    """Heuristic to extract the base franchise name (e.g. 'Toy Story 3' -> 'toy story')."""
    # Remove year if present like (1995)
    t = re.sub(r'\(\d{4}\)', '', str(title)).strip()
    # Split by colon or dash (e.g. "Avengers: Age of Ultron" -> "Avengers")
    t = re.split(r'[:\-]', t)[0].strip()
    # Remove trailing numbers or simple Roman numerals
    t = re.sub(r'\s+\d+$', '', t).strip()
    t = re.sub(r'\s+[IVX]+$', '', t).strip()
    return t.lower()

#TWO-STAGE Recommendation Pipeline ───────────────────────────────────

@torch.no_grad()
def recommend(rated_movies, top_n=20, stage1_candidates=100):
    _load() # Ensures both models are loaded
    
    # Mappings and Metadata
    s1_mid2idx = _bundle['movie_id_to_idx']
    s1_idx2mid = _bundle['idx_to_movie_id']
    
    # DO NOT overwrite s2_mid2idx and s2_idx2mid here! 
    # They are correctly using the globally loaded sasrec_mappings.
    global s2_mid2idx, s2_idx2mid
    
    meta = _bundle['movies_meta'].set_index('MovieID')
    
    assert len(rated_movies) == 15, 'Exactly 15 ratings required'

    # Prepare inputs from user's rated movies
    seen_mids = {int(e['movie_id']) for e in rated_movies}
    liked_idxs_s1 = [s1_mid2idx[int(e['movie_id'])] for e in rated_movies if float(e['rating']) >= 4.0 and int(e['movie_id']) in s1_mid2idx]
    if not liked_idxs_s1: # Fallback if no high ratings
        liked_idxs_s1 = [s1_mid2idx[int(e['movie_id'])] for e in rated_movies if int(e['movie_id']) in s1_mid2idx]

    # STAGE 1: CANDIDATE GENERATION (TWO-TOWER)
    # Generate a user embedding based on their liked items
    user_history_tensor = torch.tensor(liked_idxs_s1, dtype=torch.long).to(_device)
    liked_item_reps = _model.get_item_rep(user_history_tensor)
    dynamic_user_embedding = liked_item_reps.mean(dim=0, keepdim=True)
    
    # Get scores for all items and find the top candidates
    raw_scores = torch.matmul(dynamic_user_embedding, _all_item_representations.T).squeeze()
    
    top_candidate_indices = torch.topk(raw_scores, k=stage1_candidates + len(seen_mids)).indices
    
    candidate_mids = []
    for idx in top_candidate_indices.tolist():
        mid = s1_idx2mid[idx]
        if mid not in seen_mids:
            candidate_mids.append(mid)
    candidate_mids = candidate_mids[:stage1_candidates]

    # ---------------------------------------------------------
    # STAGE 2: SHORT-TERM INTENT (SASRec Top 5)
    # ---------------------------------------------------------
    MAX_SEQ_LENGTH = _sasrec_model.max_seq_length
    
    # 1. Extract ONLY the last five 5-star movies
    five_star_movies = [e for e in rated_movies if float(e['rating']) == 5.0]
    last_five_5star = five_star_movies[-5:] if len(five_star_movies) >= 5 else five_star_movies
    sasrec_sequence_mids = [int(e['movie_id']) for e in last_five_5star]
    
    # 2. Convert to indices and pad (NO MANUAL +1 SHIFT NEEDED)
    seq_indices = []
    for mid in sasrec_sequence_mids:
        if mid in s2_mid2idx:
            # We removed the +1 here because s2_mid2idx is ALREADY 1-indexed
            seq_indices.append(s2_mid2idx[mid]) 
            
    padded_seq = ([0] * (MAX_SEQ_LENGTH - len(seq_indices))) + seq_indices
    padded_seq = padded_seq[-MAX_SEQ_LENGTH:]
    
    input_tensor = torch.tensor([padded_seq], dtype=torch.long, device=_device)
    
    # 3. Get predictions from SASRec for all items
    logits = _sasrec_model(input_tensor)
    sasrec_scores = logits[0] # Logits for all items based on the sequence
    
    # Get top 50 to filter out seen movies and secure 5 valid recommendations
    top_sasrec_indices = torch.topk(sasrec_scores, k=50).indices.tolist()
    
    sasrec_top_mids = []
    for idx in top_sasrec_indices:
        if idx == 0: 
            continue # Skip the padding token if the model predicts it
            
        # We removed the -1 here. `idx` directly maps to the movie ID in `s2_idx2mid`
        if idx in s2_idx2mid: 
            mid = s2_idx2mid[idx]
            if mid not in seen_mids:
                sasrec_top_mids.append(mid)
                if len(sasrec_top_mids) == 5:
                    break


    # ---------------------------------------------------------
    # 3. BLEND THE RESULTS (Sequel Boost -> Top 5 SASRec -> Two-Tower)
    # ---------------------------------------------------------
    final_recommendation_mids = []
    
    # A. Identify the franchise of the VERY LAST 5-star movie
    base_franchise = None
    if last_five_5star:
        last_5star_mid = int(last_five_5star[-1]['movie_id'])
        if last_5star_mid in meta.index:
            last_title = meta.loc[last_5star_mid]['Title_Clean']
            base_franchise = get_base_franchise(last_title)
    
    # B. Boost Sequels: Look through ALL candidates for a title match
    if base_franchise and len(base_franchise) >= 4: # >4 avoids matching short generic words
        all_candidates = sasrec_top_mids + candidate_mids
        for mid in all_candidates:
            if mid not in final_recommendation_mids and mid in meta.index:
                cand_title = str(meta.loc[mid]['Title_Clean']).lower()
                if base_franchise in cand_title:
                    final_recommendation_mids.append(mid)

    # C. Add top 5 from SASRec (if not already added as a sequel)
    for mid in sasrec_top_mids:
        if mid not in final_recommendation_mids:
            final_recommendation_mids.append(mid)
            
    # D. Fill the rest with Two-Tower candidates (up to top_n)
    for mid in candidate_mids:
        if len(final_recommendation_mids) >= top_n:
            break
        if mid not in final_recommendation_mids:
            final_recommendation_mids.append(mid)


    # ---------------------------------------------------------
    # 4. FORMAT FINAL OUTPUT
    # ---------------------------------------------------------
    def parse_year_25m(row):
        try: return int(float(str(row.get('Year', ''))))
        except (ValueError, TypeError): return None
        
    results = []
    for i, mid in enumerate(final_recommendation_mids):
        # Add this check to skip movie IDs that aren't in the metadata
        if mid not in meta.index:
            continue 
            
        row = meta.loc[mid]
        results.append({
            'movie_id'        : int(mid),
            'title'           : str(row['Title_Clean']),
            'genres'          : str(row.get('Genres', '')),
            'year'            : parse_year_25m(row),
            'predicted_rating': round(float(top_n - i), 4), # Dummy descending score for frontend ranking
        })
        
        # Stop once we have exactly top_n results (since we might have skipped some)
        if len(results) == top_n:
            break
        
    return results


