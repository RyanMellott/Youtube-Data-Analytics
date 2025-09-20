import re, math, numpy as np, pandas as pd, torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import median_abs_deviation
# Load CSV 

path = r'C:\Users\rrmel\Downloads\Youtube Data.csv'
df = pd.read_csv(
    path,
    encoding='latin-1',
    sep=',',
    quotechar='"',
    doublequote=True,
    engine='python',
)
# Normalize header names
df.columns = (
    df.columns.str.replace('\ufeff','', regex=False).str.strip().str.lower()
)

# Find the category column and channel column
cat_candidates  = ('category', 'category_name', 'categories', 'category_id')
chan_candidates = ('channel_name', 'channel_title', 'channel')

cat_col  = next((c for c in cat_candidates  if c in df.columns), None)
chan_col = next((c for c in chan_candidates if c in df.columns), None)

if cat_col is None:
    raise KeyError(f"No category-like column found. Columns are: {df.columns.tolist()}")
if chan_col is None:
    raise KeyError(f"No channel-like column found. Columns are: {df.columns.tolist()}")

# Rename 
rename_map = {}
if cat_col  != 'category':     rename_map[cat_col]  = 'category'
if chan_col != 'channel_name': rename_map[chan_col] = 'channel_name'
if rename_map:
    df = df.rename(columns=rename_map)

df['category']     = df['category'].astype('string').fillna('Unknown')
df['channel_name'] = df['channel_name'].astype('string').fillna('Unknown')

# Cleaning data

print(df.columns.tolist())
print(list(map(repr, df.columns))) 

# 2) Normalize headers
df.columns = (
    df.columns
      .str.replace('\ufeff', '', regex=False) 
      .str.strip()
      .str.lower()
)

# 3) Pick the best match for a category-like column
candidates = ('category', 'category_name', 'categories')
cat_col = next((c for c in candidates if c in df.columns), None)

if cat_col is None:
    raise KeyError(f"No category-like column found. Columns are: {df.columns.tolist()}")

df[cat_col] = df[cat_col].astype('string').fillna('Unknown')


target_col = 'view_count'   # or 'like_count'

df['video_id']     = df['video_id'].astype('category')
df['title']        = df['title'].fillna('').astype(str)
df['tags']         = df['tags'].fillna('No Tag').astype(str)   
df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
df['category']    = df['category'].fillna('Unknown').astype(str)
df = df[df[target_col].notna()]
df['y'] = np.log1p(df[target_col].astype(float))

# Vocab for text

word_pat = re.compile(r"[A-Za-z0-9_]+")

def tokenize(s: str):
    return word_pat.findall(s.lower())

from collections import Counter
cnt = Counter()
for t in df['title']:
    cnt.update(tokenize(t))
for tg in df['tags']:
    cnt.update(tokenize(tg))

max_vocab = 30000
itos = ['<pad>', '<unk>'] + [w for w,_ in cnt.most_common(max_vocab-2)]
stoi = {w:i for i,w in enumerate(itos)}
PAD, UNK = 0, 1

def numericalize(tokens):
    return [stoi.get(w, UNK) for w in tokens]

# Categorical indexers

def make_indexer(series):
    uniques = ['<pad>'] + sorted(series.unique().tolist())
    to_idx = {k:i for i,k in enumerate(uniques)}
    return to_idx, uniques

chan_to_idx, _ = make_indexer(df['channel_name'])
cat_to_idx,  _ = make_indexer(df['category'])

# Dataset

class YouTubeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.f = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, idx):
        row = self.f.iloc[idx]

        title_ids = numericalize(tokenize(row['title']))
        tags_ids  = numericalize(tokenize(row['tags']))
        ch_id     = chan_to_idx.get(row['channel_name'], 0)
        cat_id    = cat_to_idx.get(row['category'], 0)

        n_title = len(title_ids)
        raw_tag_list = re.split(r'[|,;/]+', row['tags']) if row['tags'] else []
        n_tags  = len([t for t in raw_tag_list if t.strip()]) if row['tags'] != 'No Tag' else 0

        x_num = np.array([n_title, n_tags], dtype=np.float32)

        return {
            'title_ids': torch.tensor(title_ids or [PAD], dtype=torch.long),
            'tags_ids':  torch.tensor(tags_ids  or [PAD], dtype=torch.long),
            'ch_id':     torch.tensor(ch_id, dtype=torch.long),
            'cat_id':    torch.tensor(cat_id, dtype=torch.long),
            'x_num':     torch.tensor(x_num, dtype=torch.float32),
            'y':         torch.tensor(row['y'], dtype=torch.float32),
        }


def pad_sequence_1d(seqs, pad_value=0):
    maxlen = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_value, dtype=seqs[0].dtype)
    for i,s in enumerate(seqs):
        out[i, :s.size(0)] = s
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    return out, lengths

def collate(batch):
    title_ids = [b['title_ids'] for b in batch]
    tags_ids  = [b['tags_ids']  for b in batch]
    title_pad, title_len = pad_sequence_1d(title_ids, PAD)
    tags_pad,  tags_len  = pad_sequence_1d(tags_ids,  PAD)
    ch   = torch.stack([b['ch_id'] for b in batch])
    cat  = torch.stack([b['cat_id'] for b in batch])
    xnum = torch.stack([b['x_num'] for b in batch])
    y    = torch.stack([b['y'] for b in batch])
    return (title_pad, title_len, tags_pad, tags_len, ch, cat, xnum), y


# Split & loaders (small dataset: smaller batch helps)

tr, te = train_test_split(df, test_size=0.2, random_state=42)
ds_tr, ds_te = YouTubeDataset(tr), YouTubeDataset(te)
dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True,  collate_fn=collate)
dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, collate_fn=collate)

# Model

class MeanPoolEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)

    def forward(self, ids, lengths):
        E = self.emb(ids)                     
        mask = (ids != PAD).unsqueeze(-1)     
        summed = (E * mask).sum(dim=1)        
        denom = lengths.clamp(min=1).unsqueeze(-1).float()
        return summed / denom

class YouTubeRegressor(nn.Module):
    def __init__(self, vocab_size, emb_size=64, n_channels=1, n_categories=1, n_numeric=2, hidden_size=128):
        super().__init__()
        self.title_emb = MeanPoolEmbedding(vocab_size, emb_size, PAD)
        self.tags_emb  = MeanPoolEmbedding(vocab_size, emb_size, PAD)
        self.ch_emb    = nn.Embedding(n_channels, emb_size, padding_idx=0)
        self.cat_emb   = nn.Embedding(n_categories, emb_size, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size*4 + n_numeric, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, title_ids, title_len, tags_ids, tags_len, ch_id, cat_id, x_num):
        title_vec = self.title_emb(title_ids, title_len)
        tags_vec  = self.tags_emb(tags_ids, tags_len)
        ch_vec    = self.ch_emb(ch_id)
        cat_vec   = self.cat_emb(cat_id)
        x = torch.cat([title_vec, tags_vec, ch_vec, cat_vec, x_num], dim=1)
        return self.mlp(x).squeeze(1)   

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YouTubeRegressor(
    vocab_size=len(itos),
    emb_size=64,
    n_channels=len(chan_to_idx),
    n_categories=len(cat_to_idx),
    n_numeric=2,
    hidden_size=128
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

def rmse_log(y_true, y_pred):
    return math.sqrt(((y_true - y_pred)**2).mean().item())

# Train

epochs = 12  
for epoch in range(epochs):
    model.train()
    total = 0.0
    for (X, y) in dl_tr:
        X = [x.to(device) for x in X]
        y = y.to(device)
        pred = model(*X)
        loss = loss_fn(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * y.size(0)
    train_mse = total / len(ds_tr)

# Evaluation
    model.eval()
    with torch.no_grad():
        ys, ps = [], []
        for (X, y) in dl_te:
            X = [x.to(device) for x in X]
            y = y.to(device)
            p = model(*X)
            ys.append(y); ps.append(p)
        y_all = torch.cat(ys); p_all = torch.cat(ps)
        val_rmse_log = rmse_log(y_all, p_all)
        val_rmse_orig = math.sqrt(((torch.expm1(y_all) - torch.expm1(p_all))**2).mean().item())

    print(f"Epoch {epoch+1:02d} | train MSE={train_mse:.4f} | val RMSE(log)={val_rmse_log:.4f} | val RMSE(orig)={val_rmse_orig:.2f}")

# Predict helper

def predict_on_dataframe(frame: pd.DataFrame):
    ds = YouTubeDataset(frame)
    dl = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    model.eval()
    outs = []
    with torch.no_grad():
        for (X, _) in dl:
            X = [x.to(device) for x in X]
            pred_log = model(*X)
            outs.append(torch.expm1(pred_log).cpu().numpy())  
    return np.concatenate(outs)

# Summary Data
numeric_candidates = ['view_count', 'like_count', 'comment_count']
for c in numeric_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

def parse_duration_iso8601(s):
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', str(s))
    if not m:
        return np.nan
    h = int(m.group(1) or 0)
    mnt = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    return h*3600 + mnt*60 + sec

if 'duration' in df.columns:
    df['duration_sec'] = df['duration'].apply(parse_duration_iso8601)

metrics = []

if 'view_count' in df.columns:
    metrics += [
        ('Mean Views',   df['view_count'].mean()),
        ('Median Views', df['view_count'].median()),
        ('SD Views',     df['view_count'].std()),
        ('MAD Views',    median_abs_deviation(df['view_count'].dropna())),
    ]

if metrics:
    labels, values = zip(*metrics)
    plt.figure(figsize=(8,4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns available to summarize.")
