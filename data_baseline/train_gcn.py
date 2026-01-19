import textwrap, os, pathlib

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import (

    GINEConv, TransformerConv,

    global_mean_pool, global_max_pool, global_add_pool,
    AttentionalAggregation, Set2Set,
    JumpingKnowledge,
)

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn
)

# Data loader

BASE = os.path.expanduser("/home/onyxia/work/DataChallengeAltegrad/data_baseline/")

TRAIN_GRAPHS = os.path.join(BASE, "train_graphs.pkl")
VAL_GRAPHS   = os.path.join(BASE, "validation_graphs.pkl")
TEST_GRAPHS  = os.path.join(BASE, "test_graphs.pkl")

TRAIN_EMB_CSV = os.path.join("/home/onyxia/work/DataChallengeAltegrad/data_baseline/train_embeddings_ensemble_chimberta_gte_large.csv")
VAL_EMB_CSV   = os.path.join("home/onyxia/work/DataChallengeAltegrad/data_baseline/validation_embeddings_ensemble_chimberta_gte_large.csv")

# Configuration 

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model parameters
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = 0.1
USE_EDGE_FEATURES = True

# Architecture possibilites (simple knobs)
ARCH = "gine"          
POOL = "multipool"    
JK_MODE = "last"      

# Modelling components

class ResidualMLP(nn.Module):
    """Cheap stabilizer for the projection head."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ff(self.norm(x))

class AtomFeatureEncoder(nn.Module):
    """
    We using the nodes features that were not taken into account by the first architecture.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(119, hidden),  
            nn.Embedding(10, hidden),   
            nn.Embedding(11, hidden),   
            nn.Embedding(12, hidden),   
            nn.Embedding(9, hidden),    
            nn.Embedding(5, hidden),    
            nn.Embedding(8, hidden),   
            nn.Embedding(2, hidden),  
            nn.Embedding(2, hidden),    
        ])
        self.proj = nn.Linear(hidden * len(self.embeds), hidden)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        return self.proj(torch.cat(parts, dim=-1))


class BondFeatureEncoder(nn.Module):
    """
    We using the edges features that were not taken into account by the first architecture.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(23, hidden),  
            nn.Embedding(7, hidden),  
            nn.Embedding(2, hidden),   
        ])
        self.proj = nn.Linear(hidden * len(self.embeds), hidden)

    def forward(self, e_cat: torch.Tensor) -> torch.Tensor:
        parts = [emb(e_cat[:, i]) for i, emb in enumerate(self.embeds)]
        return self.proj(torch.cat(parts, dim=-1))

# Pooling options 

class MultiPool(nn.Module):
    """Mean + Max + Add pooling with small post-layer."""
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.out_dim = hidden * 3
        self.post = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        hm = global_mean_pool(h, batch_vec)
        hM = global_max_pool(h, batch_vec)
        ha = global_add_pool(h, batch_vec)
        return self.post(torch.cat([hm, hM, ha], dim=-1))


class AttentionPool(nn.Module):
    """AttentionalAggregation pooling (often a small gain, low complexity)."""
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.pool = AttentionalAggregation(gate_nn=gate)
        self.out_dim = hidden

    def forward(self, h: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        return self.pool(h, batch_vec)


class Set2SetPool(nn.Module):
    """Set2Set pooling (a bit heavier than attn, still reasonable)."""
    def __init__(self, hidden: int, steps: int = 3):
        super().__init__()
        self.pool = Set2Set(hidden, processing_steps=steps)
        self.out_dim = hidden * 2

    def forward(self, h: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        return self.pool(h, batch_vec)


# New encoder model

class ImprovedMolGNN(nn.Module):

    def __init__(
        self,
        hidden: int = 256,
        out_dim: int = 768,
        layers: int = 4,
        dropout: float = 0.1,
        use_edge_feat: bool = True,
        arch: str = "gine",
        pool: str = "multipool",
        jk: str = "last",
        heads: int = 4,  # used only for transformer
    ):
        super().__init__()
        self.use_edge_feat = use_edge_feat
        self.arch = arch.lower().strip()
        self.dropout = dropout

        self.atom_enc = AtomFeatureEncoder(hidden)

        self.bond_enc = BondFeatureEncoder(hidden) if use_edge_feat else None

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.arch == "transformer":
        
            assert hidden % heads == 0, "hidden must be divisible by heads"
            for _ in range(layers):
                self.convs.append(
                    TransformerConv(
                        in_channels=hidden,
                        out_channels=hidden // heads,
                        heads=heads,
                        dropout=dropout,
                        edge_dim=hidden if use_edge_feat else None,
                        beta=True,
                    )
                )
                self.norms.append(nn.LayerNorm(hidden))
        else:
            
            for _ in range(layers):
                nn_msg = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINEConv(nn_msg, edge_dim=hidden if use_edge_feat else None))
                self.norms.append(nn.LayerNorm(hidden))

        # Jumping Knowledge
        self.jk_mode = jk
        self.jk = JumpingKnowledge(jk) if jk in ("cat", "max") else None
        self.layers = layers
        self.hidden = hidden

        # Pooling
        pool = pool.lower().strip()
        if pool == "attn":
            self.pool = AttentionPool(hidden, dropout)
        elif pool == "set2set":
            self.pool = Set2SetPool(hidden, steps=3)
        else:
            self.pool = MultiPool(hidden, dropout)

        pooled_dim = self.pool.out_dim
        if self.jk is not None and self.jk_mode == "cat":
            if isinstance(self.pool, MultiPool):
                pooled_dim = (hidden * layers) * 3
            elif isinstance(self.pool, Set2SetPool):
                pooled_dim = (hidden * layers) * 2
            else:  
                pooled_dim = (hidden * layers)

        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(hidden * 2, dropout),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:

        h = self.atom_enc(batch.x)

        e = None
        if self.use_edge_feat and getattr(batch, "edge_attr", None) is not None:
            e = self.bond_enc(batch.edge_attr)

        hs = []
        for conv, norm in zip(self.convs, self.norms):
            if self.arch == "transformer":

                h = conv(h, batch.edge_index, e) if e is not None else conv(h, batch.edge_index)
            else:
                h = conv(h, batch.edge_index, e) if e is not None else conv(h, batch.edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)

        if self.jk is None or self.jk_mode == "last":
            h_out = hs[-1]
        else:
            h_out = self.jk(hs)

        g = self.pool(h_out, batch.batch)
        out = self.projection(g)
        return F.normalize(out, dim=-1)


# LOos s function

class ImprovedContrastiveLoss(nn.Module):
    """
    We create a new contrastive loss to replace the old MSE, it is better suited in our case.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, mol_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        bs = mol_emb.size(0)
        mol_emb = F.normalize(mol_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        logits = (mol_emb @ txt_emb.t()) / self.temperature
        labels = torch.arange(bs, device=mol_emb.device)

        loss_m2t = F.cross_entropy(logits, labels)
        loss_t2m = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_m2t + loss_t2m)


# Training and evaluation 

def train_epoch(mol_enc, loader, optimizer, criterion, device, scheduler=None):
    mol_enc.train()
    total_loss, total = 0.0, 0

    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        loss = criterion(mol_vec, text_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mol_enc.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    mol_enc.eval()

    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))

    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    correct = torch.arange(N, device=device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()
    results = {"MRR": mrr}
    for k in (1, 5, 10):
        results[f"Hit@{k}"] = (pos <= k).float().mean().item()
    return results



def main():
    print(f"Device: {DEVICE}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

    emb_dim = len(next(iter(train_emb.values())))
    print(f"Text embedding dimension: {emb_dim}")

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        print("Please run: python prepare_graph_data.py")
        return

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    mol_enc = ImprovedMolGNN(
        hidden=HIDDEN_DIM,
        out_dim=emb_dim,
        layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_edge_feat=USE_EDGE_FEATURES,
        arch=ARCH,
        pool=POOL,
        jk=JK_MODE,
        heads=4,
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in mol_enc.parameters()):,}")
    print(f"Arch={ARCH} | Pool={POOL} | JK={JK_MODE} | EdgeFeat={USE_EDGE_FEATURES}")

    criterion = ImprovedContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_dl) * EPOCHS
    warmup_steps = len(train_dl) * 2
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    # Validation part, we measure MRR

    best_mrr = 0.0
    patience = 5
    patience_counter = 0

    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, criterion, DEVICE, scheduler)

        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
            current_mrr = val_scores.get("MRR", 0.0)

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                patience_counter = 0
                checkpoint = {
                    'model_state_dict': mol_enc.state_dict(),
                    'config': {
                        'hidden': HIDDEN_DIM,
                        'out_dim': emb_dim,
                        'layers': NUM_LAYERS,
                        'dropout': DROPOUT,
                        'use_edge_feat': USE_EDGE_FEATURES,
                        'arch': ARCH,
                        'pool': POOL,
                        'jk': JK_MODE,
                        'heads': 4,
                    }
                }
                torch.save(checkpoint, "best_model.pt")

                print(f"New best model saved (MRR: {best_mrr:.4f})")
            else:
                patience_counter += 1

            print(
                f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - "
                f"MRR={current_mrr:.4f} - Hit@1={val_scores.get('Hit@1', 0):.4f} - "
                f"Hit@5={val_scores.get('Hit@5', 0):.4f}"
            )

            if patience_counter >= patience:
                print(f"\nEarly stopping after {ep+1} epochs")
                break
        else:
            print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f}")

    model_path = f"model_{ARCH}_{POOL}_{JK_MODE}_output.pt"
    checkpoint = {
                    'model_state_dict': mol_enc.state_dict(),
                    'config': {
                        'hidden': HIDDEN_DIM,
                        'out_dim': emb_dim,
                        'layers': NUM_LAYERS,
                        'dropout': DROPOUT,
                        'use_edge_feat': USE_EDGE_FEATURES,
                        'arch': ARCH,
                        'pool': POOL,
                        'jk': JK_MODE,
                        'heads': 4,
                    }
                }
    torch.save(checkpoint, model_path)
    print(f"\nFinal model saved to {model_path}")
    print(f"Best validation MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()