import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

from train_gcn import (
    ImprovedMolGNN, DEVICE
)


BASE = os.path.expanduser("~/Desktop/ENSAE/AL for texts and graphs/DataChallengeAltegrad/data")
base_antoine = os.path.expanduser("~/DataChallengeAltegrad")

TRAIN_GRAPHS = os.path.join(BASE, "train_graphs.pkl")
VAL_GRAPHS = os.path.join(BASE, "validation_graphs.pkl")
TEST_GRAPHS = os.path.join(BASE, "test_graphs.pkl")

TRAIN_EMB_CSV = os.path.join(BASE, "train_embeddings_mpnet_mean_normalize.csv")
VAL_EMB_CSV = os.path.join(BASE, "validation_embeddings_mpnet_mean_normalize.csv")


def top_k_retrieval_strategies():
    """Return available top-K retrieval strategies."""
    return {
        'top1': 'Single best match',
        'topk_mean': 'Mean of top-K embeddings',
        'topk_weighted': 'Weighted mean by similarity',
        'topk_most_similar': 'Most similar among top-K (reranking)',
        'topk_diverse': 'Diverse sampling from top-K',
        'ensemble': 'Ensemble of multiple strategies'
    }


@torch.no_grad()
def retrieve_top_k_descriptions(model, train_data, test_data, train_emb_dict, device, 
                               output_csv, top_k=5, strategy='topk_weighted', 
                               temperature=0.1, diversity_weight=0.3):
    """
    Args:
        model: Trained GNN model
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings
        device: Device to run on
        output_csv: Path to save retrieved descriptions
        top_k: Number of nearest neighbors to retrieve
        strategy: Retrieval strategy ('top1', 'topk_mean', 'topk_weighted', 
                  'topk_most_similar', 'topk_diverse', 'ensemble')
        temperature: Temperature for softmax weighting (for weighted strategies)
        diversity_weight: Weight for diversity in 'topk_diverse' strategy
    """
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    print(f"Using top-K retrieval with k={top_k}, strategy='{strategy}'")
    
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
    # Encode test molecules
    test_mol_embs = []
    test_ids_ordered = []
    
    print("Encoding test molecules...")
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    test_mol_embs = F.normalize(test_mol_embs, dim=-1)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    # Compute similarities in batches to save memory
    print(f"Computing similarities with {len(train_ids)} training samples...")
    batch_size = 1024
    similarities = []
    
    for i in range(0, len(test_mol_embs), batch_size):
        batch_embs = test_mol_embs[i:i+batch_size]
        batch_sim = batch_embs @ train_embs.t()
        similarities.append(batch_sim.cpu())
    
    similarities = torch.cat(similarities, dim=0)
    
    # Get top-K indices and values
    top_k_values, top_k_indices = torch.topk(similarities, k=min(top_k, len(train_ids)), dim=-1)
    
    results = []
    all_top_k_info = []  # Store all top-K info for analysis
    
    print(f"\nRetrieving descriptions using strategy: {strategy}")
    
    for i, test_id in enumerate(test_ids_ordered):
        top_k_idx = top_k_indices[i]  # Shape: [top_k]
        top_k_sim = top_k_values[i]   # Shape: [top_k]
        
        # Convert to lists for easier handling
        top_k_idx_list = top_k_idx.tolist()
        top_k_sim_list = top_k_sim.tolist()
        top_k_ids = [train_ids[idx] for idx in top_k_idx_list]
        top_k_descs = [train_id2desc[train_id] for train_id in top_k_ids]
        
        # Store top-K info for analysis
        all_top_k_info.append({
            'test_id': test_id,
            'top_k_ids': top_k_ids,
            'top_k_similarities': top_k_sim_list,
            'top_k_descriptions': top_k_descs
        })
        
        # Apply retrieval strategy
        if strategy == 'top1':
            # Strategy 1: Simple top-1 (baseline)
            retrieved_idx = top_k_idx_list[0]
            retrieved_desc = top_k_descs[0]
            
        elif strategy == 'topk_mean':
            # Strategy 2: Mean of top-K embeddings
            # Get embeddings of top-K
            top_k_embeddings = train_embs[top_k_idx].cpu()  # [k, emb_dim]
            # Compute mean embedding
            mean_embedding = top_k_embeddings.mean(dim=0, keepdim=True)  # [1, emb_dim]
            mean_embedding = F.normalize(mean_embedding, dim=-1)
            
            # Find most similar to mean among top-K
            sim_to_mean = mean_embedding @ top_k_embeddings.t()  # [1, k]
            best_in_topk = torch.argmax(sim_to_mean).item()
            retrieved_idx = top_k_idx_list[best_in_topk]
            retrieved_desc = top_k_descs[best_in_topk]
            
        elif strategy == 'topk_weighted':
            # Strategy 3: Weighted mean by similarity scores
            weights = F.softmax(top_k_sim / temperature, dim=-1).cpu()  # [k]
            top_k_embeddings = train_embs[top_k_idx].cpu()  # [k, emb_dim]
            
            # Weighted average
            weighted_embedding = torch.sum(top_k_embeddings * weights.unsqueeze(-1), dim=0, keepdim=True)
            weighted_embedding = F.normalize(weighted_embedding, dim=-1)
            
            # Find most similar to weighted embedding among top-K
            sim_to_weighted = weighted_embedding @ top_k_embeddings.t()
            best_in_topk = torch.argmax(sim_to_weighted).item()
            retrieved_idx = top_k_idx_list[best_in_topk]
            retrieved_desc = top_k_descs[best_in_topk]
            
        elif strategy == 'topk_most_similar':
            # Strategy 4: Reranking - find most similar pair among top-K
            # This helps when the most similar might have high similarity to query
            # but might not be the best "representative" of the cluster
            retrieved_idx = top_k_idx_list[0]  # Default to top-1
            retrieved_desc = top_k_descs[0]
            
        elif strategy == 'topk_diverse':
            # Strategy 5: Diversity-promoting selection
            # Try to select a description that is somewhat different from others
            # to avoid always picking the same type of description
            top_k_embeddings = train_embs[top_k_idx].cpu()  # [k, emb_dim]
            
            # Compute pairwise similarities among top-K
            pairwise_sim = top_k_embeddings @ top_k_embeddings.t()  # [k, k]
            
            # For each candidate, compute average similarity to others
            avg_similarity_to_others = []
            for j in range(len(top_k_embeddings)):
                # Exclude self-similarity
                others_sim = torch.cat([pairwise_sim[j, :j], pairwise_sim[j, j+1:]])
                avg_sim = others_sim.mean()
                avg_similarity_to_others.append(avg_sim.item())
            
            # Convert to diversity scores (lower average similarity = more diverse)
            diversity_scores = 1 - torch.tensor(avg_similarity_to_others)
            
            # Combined score: similarity to query + diversity
            combined_scores = top_k_sim.cpu() + diversity_weight * diversity_scores
            
            # Select best based on combined score
            best_idx = torch.argmax(combined_scores).item()
            retrieved_idx = top_k_idx_list[best_idx]
            retrieved_desc = top_k_descs[best_idx]
            
        elif strategy == 'ensemble':
            # Strategy 6: Ensemble of multiple strategies
            strategies_to_try = ['top1', 'topk_mean', 'topk_weighted']
            candidate_descs = []
            
            for strat in strategies_to_try:
                # Temporarily use this strategy
                if strat == 'top1':
                    candidate_descs.append(top_k_descs[0])
                elif strat == 'topk_mean':
                    top_k_embeddings = train_embs[top_k_idx].cpu()
                    mean_embedding = top_k_embeddings.mean(dim=0, keepdim=True)
                    mean_embedding = F.normalize(mean_embedding, dim=-1)
                    sim_to_mean = mean_embedding @ top_k_embeddings.t()
                    best_in_topk = torch.argmax(sim_to_mean).item()
                    candidate_descs.append(top_k_descs[best_in_topk])
                elif strat == 'topk_weighted':
                    weights = F.softmax(top_k_sim / temperature, dim=-1).cpu()
                    top_k_embeddings = train_embs[top_k_idx].cpu()
                    weighted_embedding = torch.sum(top_k_embeddings * weights.unsqueeze(-1), dim=0, keepdim=True)
                    weighted_embedding = F.normalize(weighted_embedding, dim=-1)
                    sim_to_weighted = weighted_embedding @ top_k_embeddings.t()
                    best_in_topk = torch.argmax(sim_to_weighted).item()
                    candidate_descs.append(top_k_descs[best_in_topk])
            
            from collections import Counter
            desc_counter = Counter(candidate_descs)
            retrieved_desc = desc_counter.most_common(1)[0][0]

            retrieved_idx = None
            for idx, desc in enumerate(top_k_descs):
                if desc == retrieved_desc:
                    retrieved_idx = top_k_idx_list[idx]
                    break
            if retrieved_idx is None:
                retrieved_idx = top_k_idx_list[0]
                
        else:
            
            retrieved_idx = top_k_idx_list[0]
            retrieved_desc = top_k_descs[0]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc,
            'retrieved_train_id': train_ids[retrieved_idx],
            'similarity_score': top_k_sim_list[0],  
            'top_k_used': top_k,
            'strategy': strategy
        })
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)    
    
    print(f"\nResults saved to: {output_csv}")
    
    return results_df, all_top_k_info


def main():
    print(f"Device: {DEVICE}")

    TOP_K = 3  # Number of nearest neighbors to retrieve
    STRATEGY = 'topk_weighted'  # Retrieval strategy
    TEMPERATURE = 0.2  # For softmax weighting
    OUTPUT_CSV = f"test_retrieved_k{TOP_K}_{STRATEGY}_final.csv"
    
    model_path = "model_long_transformers_ne.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gcn.py")
        return
    
    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TEST_GRAPHS}")
        return
    
    # Load text embeddings
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    emb_dim = len(next(iter(train_emb.values())))

    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)

    config = checkpoint.get('config', {})
    

    model = ImprovedMolGNN(
        hidden=config.get('hidden', 256),
        out_dim=config.get('out_dim', emb_dim),
        layers=config.get('layers', 4),
        dropout=config.get('dropout', 0.1),
        use_edge_feat=config.get('use_edge_feat', True),
        arch=config.get('arch', 'transformer'), 
        pool=config.get('pool', 'multipool'),
        jk=config.get('jk', 'last'),
        heads=config.get('heads', 4)
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\n" + "="*80)
    print("TOP-K NEAREST NEIGHBOR RETRIEVAL")
    print("="*80)
    print(f"\nSelected: k={TOP_K}, strategy='{STRATEGY}'")
    print("="*80)
    
    # Single strategy retrieval
    results_df, top_k_info = retrieve_top_k_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=OUTPUT_CSV,
        top_k=TOP_K,
        strategy=STRATEGY,
        temperature=TEMPERATURE
    )


if __name__ == "__main__":
    main()