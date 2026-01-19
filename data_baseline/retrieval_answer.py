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

# data loader

BASE = os.path.expanduser("~/Desktop/ENSAE/AL for texts and graphs/DataChallengeAltegrad/data")
# BASE = os.path.expanduser("/home/onyxia/work/DataChallengeAltegrad/data_baseline/")

TRAIN_GRAPHS = os.path.join(BASE, "train_graphs.pkl")
VAL_GRAPHS = os.path.join(BASE, "validation_graphs.pkl")
TEST_GRAPHS = os.path.join(BASE, "test_graphs.pkl")

TRAIN_EMB_CSV = os.path.join(BASE, "train_embeddings_ensemble_chimberta_gte_large.csv")
VAL_EMB_CSV = os.path.join(BASE, "validation_embeddings_ensemble_chimberta_gte_large.csv")


def top_k_retrieval_strategies():
    """Return available top-K retrieval strategies."""
    return {
        'top1': 'Single best match',
        'topk_mean': 'Mean of top-K embeddings',
        'topk_weighted': 'Weighted mean by similarity'
    }


@torch.no_grad()
def retrieve_top_k_descriptions(model, train_data, test_data, train_emb_dict, device, 
                               output_csv, top_k=5, strategy='topk_weighted', 
                               temperature=0.1, diversity_weight=0.3):
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    print(f"Using top-K retrieval with k={top_k}, strategy='{strategy}'")
    
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
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

    batch_size = 1024
    similarities = []
    
    for i in range(0, len(test_mol_embs), batch_size):
        batch_embs = test_mol_embs[i:i+batch_size]
        batch_sim = batch_embs @ train_embs.t()
        similarities.append(batch_sim.cpu())
    
    similarities = torch.cat(similarities, dim=0)
    
    top_k_values, top_k_indices = torch.topk(similarities, k=min(top_k, len(train_ids)), dim=-1)
    
    results = []
    all_top_k_info = []
    
    print(f"\nRetrieving descriptions using strategy: {strategy}")
    
    for i, test_id in enumerate(test_ids_ordered):
        top_k_idx = top_k_indices[i]  
        top_k_sim = top_k_values[i]   
        
        top_k_idx_list = top_k_idx.tolist()
        top_k_sim_list = top_k_sim.tolist()
        top_k_ids = [train_ids[idx] for idx in top_k_idx_list]
        top_k_descs = [train_id2desc[train_id] for train_id in top_k_ids]
        
        all_top_k_info.append({
            'test_id': test_id,
            'top_k_ids': top_k_ids,
            'top_k_similarities': top_k_sim_list,
            'top_k_descriptions': top_k_descs
        })
        
        # Apply retrieval strategy
        if strategy == 'top1':
            
            retrieved_idx = top_k_idx_list[0]
            retrieved_desc = top_k_descs[0]
            
        elif strategy == 'topk_mean':

            top_k_embeddings = train_embs[top_k_idx].cpu()  
       
            mean_embedding = top_k_embeddings.mean(dim=0, keepdim=True) 
            mean_embedding = F.normalize(mean_embedding, dim=-1)
            
            sim_to_mean = mean_embedding @ top_k_embeddings.t() 
            best_in_topk = torch.argmax(sim_to_mean).item()
            retrieved_idx = top_k_idx_list[best_in_topk]
            retrieved_desc = top_k_descs[best_in_topk]
            
        elif strategy == 'topk_weighted':

            weights = F.softmax(top_k_sim / temperature, dim=-1).cpu()  
            top_k_embeddings = train_embs[top_k_idx].cpu()  
            
            weighted_embedding = torch.sum(top_k_embeddings * weights.unsqueeze(-1), dim=0, keepdim=True)
            weighted_embedding = F.normalize(weighted_embedding, dim=-1)
            
            sim_to_weighted = weighted_embedding @ top_k_embeddings.t()
            best_in_topk = torch.argmax(sim_to_weighted).item()
            retrieved_idx = top_k_idx_list[best_in_topk]
            retrieved_desc = top_k_descs[best_in_topk]
                
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

    TOP_K = 3                   # Number of nearest neighbors to retrieve
    STRATEGY = 'topk_weighted'  # Retrieval strategy
    TEMPERATURE = 0.1         # For softmax weighting

    OUTPUT_CSV = f"test_retrieved_k{TOP_K}_{STRATEGY}_description.csv"
    
    model_path = "model_gine_multipool_last_output.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gcn.py")
        return
    
    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TEST_GRAPHS}")
        return

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
    print(f"\nSelected: k={TOP_K}, strategy='{STRATEGY}'")

    
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