#!/usr/bin/env python3


import pickle
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import List, Dict, Optional
import warnings
import json
warnings.filterwarnings('ignore')

# data loader
BASE = os.path.expanduser("~/work/DataChallengeAltegrad/data_baseline/")
TRAIN_GRAPHS = os.path.join(BASE, "train_graphs.pkl")
VAL_GRAPHS = os.path.join(BASE, "validation_graphs.pkl")

# List of open source models we could use for embeddings

EMBEDDING_MODELS = {
    # Best for chemistry - trained on scientific literature
    'chemberta': 'seyonec/ChemBERTa-zinc-base-v1',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'biobert': 'dmis-lab/biobert-v1.1',
    
    # Best general semantic models 
    'gte_large': 'Alibaba-NLP/gte-large-en-v1.5',  
    'gte_base': 'Alibaba-NLP/gte-base-en-v1.5',    
    'bge_large': 'BAAI/bge-large-en-v1.5',         
    'bge_base': 'BAAI/bge-base-en-v1.5',           
    
    # Sentence transformers (solid baselines)
    'mpnet': 'sentence-transformers/all-mpnet-base-v2', 
    'minilm': 'sentence-transformers/all-MiniLM-L12-v2', 
    
}

# NEW EMBEDDING SYSTEM

class AdvancedEmbedder:
    def __init__(self, 
                 model_name: str = 'gte_large',
                 device: str = 'auto',
                 use_instructions: bool = True):

        self.model_name = model_name
        self.use_instructions = use_instructions
        
        if model_name in EMBEDDING_MODELS:
            self.model_path = EMBEDDING_MODELS[model_name]
        else:
            self.model_path = model_name
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading {self.model_path}...")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path, device=str(self.device))
            self.is_sentence_transformer = True
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True  
            ).to(self.device)
            self.model.eval()
            self.is_sentence_transformer = False
            self.embedding_dim = self.model.config.hidden_size
            print(f"✓ Loaded with transformers ({self.embedding_dim}D)")
        
        print(f"Device: {self.device}")
        
        self.instruction = (
            "Represent this molecular description for retrieval: "
            if use_instructions else ""
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        if self.is_sentence_transformer:
            return self._get_sentence_transformer_embedding(text)
        else:
            return self._get_transformer_embedding(text)
    
    def _get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        full_text = self.instruction + text if self.instruction else text
        
        embedding = self.model.encode(
            full_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.cpu().numpy().flatten().astype(np.float32)
    
    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        full_text = self.instruction + text if self.instruction else text
        
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
            
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                
                embedding = sum_embeddings / sum_mask
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        
        return embedding.cpu().numpy().flatten().astype(np.float32)
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
 
        if self.is_sentence_transformer:
            
            full_texts = [self.instruction + t if self.instruction else t for t in texts]
            
            embeddings = self.model.encode(
                full_texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            return np.array(embeddings, dtype=np.float32)
        else:
      
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch = texts[i:i + batch_size]
                batch_embeddings = [self.get_embedding(text) for text in batch]
                all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings, dtype=np.float32)


class EnsembleEmbedder:
    """Combine multiple models for optimal performance."""
    
    def __init__(self, 
                 models: List[str] = None,
                 weights: List[float] = None,
                 device: str = 'auto'):
        """
        Initialize ensemble embedder.
        
        Args:
            models: List of model names to ensemble
            weights: Weights for each model (will be normalized)
            device: Device to use
        """
        if models is None:
            # Default: chemistry + best general model
            models = ['chemberta', 'gte_base']
        
        if weights is None:
            weights = [1.0] * len(models)
        
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        print(f"Initializing ensemble with {len(models)} models...")
        
        self.embedders = []
        for model_name in models:
            try:
                embedder = AdvancedEmbedder(model_name, device=device)
                self.embedders.append(embedder)
                print(f"  ✓ {model_name}")
            except Exception as e:
                print(f"  ✗ {model_name}: {e}")
        
        if not self.embedders:
            raise RuntimeError("No models loaded successfully!")

        if len(self.embedders) < len(models):
            self.weights = self.weights[:len(self.embedders)]
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        
        print(f"\n✓ Ensemble ready with {len(self.embedders)} models")
        print(f"  Weights: {[f'{w:.2f}' for w in self.weights]}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get ensemble embedding."""
        embeddings = []
        
        for embedder, weight in zip(self.embedders, self.weights):
            try:
                emb = embedder.get_embedding(text)
                embeddings.append(emb * weight)
            except Exception as e:
                print(f"Error in ensemble: {e}")
                continue
        
        if not embeddings:

            return np.zeros(768, dtype=np.float32)
        
        combined = np.concatenate(embeddings)
        
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined.astype(np.float32)
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get batch embeddings for ensemble."""
        all_embeddings = []
        
        for text in tqdm(texts, desc="Ensemble embedding"):
            emb = self.get_embedding(text)
            all_embeddings.append(emb)
        
        return np.array(all_embeddings, dtype=np.float32)


def process_dataset(split: str, embedder, save_path: str):
    """Process a dataset split and generate embeddings."""
    
    pkl_path = TRAIN_GRAPHS if split == 'train' else VAL_GRAPHS
    
    print(f"\n{'='*60}")
    print(f"Processing {split} set")
    print(f"{'='*60}")
    print(f"Loading from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)
    print(f"✓ Loaded {len(graphs)} graphs")
    

    ids = []
    descriptions = []
    
    for graph in graphs:
        ids.append(graph.id)
        desc = graph.description if hasattr(graph, 'description') else ""
        descriptions.append(desc if desc else "empty molecule")
    
    print(f"Generating embeddings...")
    embeddings = embedder.get_batch_embeddings(descriptions, batch_size=32)
    
    print(f"Saving to {save_path}...")
    embedding_strs = [','.join(map(str, emb)) for emb in embeddings]
    
    df = pd.DataFrame({
        'ID': ids,
        'embedding': embedding_strs
    })
    df.to_csv(save_path, index=False)
    
    
    metadata = {
        'split': split,
        'model': embedder.model_name if hasattr(embedder, 'model_name') else 'ensemble',
        'embedding_dim': embeddings.shape[1],
        'num_samples': len(ids)
    }
    
    metadata_path = save_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return embeddings


def main():

    STRATEGY = 'single'  
    SINGLE_MODEL = 'gte_large'  # Used if STRATEGY='single'
    ENSEMBLE_MODELS = ['chimberta', 'gte_large']  # Used if STRATEGY='ensemble'
    ENSEMBLE_WEIGHTS = [0.5, 0.5] 
    
    print(f"Strategy: {STRATEGY}")
    

    if STRATEGY == 'ensemble':
        print(f"Ensemble models: {ENSEMBLE_MODELS}")
        print(f"Ensemble weights: {ENSEMBLE_WEIGHTS}")
        embedder = EnsembleEmbedder(
            models=ENSEMBLE_MODELS,
            weights=ENSEMBLE_WEIGHTS,
            device='auto'
        )
        suffix = f"ensemble_{'_'.join(ENSEMBLE_MODELS)}"
    else:
        print(f"Single model: {SINGLE_MODEL}")
        embedder = AdvancedEmbedder(SINGLE_MODEL, device='auto')
        suffix = SINGLE_MODEL
    
    for split in ['train', 'validation']:
        save_path = os.path.join(BASE, f'{split}_embeddings_{suffix}.csv')
        process_dataset(split, embedder, save_path)
    
    print("\n" + "="*60)
    print("✓ All embeddings generated successfully!")
    print("="*60)


if __name__ == "__main__":
    
    main()