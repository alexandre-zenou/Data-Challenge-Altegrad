#!/usr/bin/env python3
"""Generate enhanced BERT embeddings for molecular descriptions with multiple strategies."""

import pickle
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import warnings
import json
warnings.filterwarnings('ignore')


MAX_TOKEN_LENGTH = 512
BASE = os.path.expanduser("~/")
TRAIN_GRAPHS = os.path.join(BASE, "train_graphs.pkl")
VAL_GRAPHS = os.path.join(BASE, "validation_graphs.pkl")


MODEL_CHOICES = {
    'bert_base': 'bert-base-uncased',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'biobert': 'dmis-lab/biobert-v1.1',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'deberta': 'microsoft/deberta-base',
    'distilbert': 'distilbert-base-uncased',  
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class EnhancedBertEmbeddings:
    """Enhanced BERT embedding generator with multiple pooling strategies."""
    
    def __init__(self, model_name: str = 'mpnet', device: str = 'auto'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the model to use (key in MODEL_CHOICES)
            device: 'cuda', 'cpu', or 'auto'
        """
        self.model_name = model_name
        self.model_path = MODEL_CHOICES.get(model_name, 'bert-base-uncased')
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading {self.model_path}...")
        
        # Handle sentence-transformers models differently
        if 'sentence-transformers' in self.model_path:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_path, device=self.device)
                self.is_sentence_transformer = True
                self.tokenizer = None  # Handled internally by SentenceTransformer
            except ImportError:
                print("Warning: sentence-transformers not installed. Falling back to transformers.")
                print("Install with: pip install sentence-transformers")
                self.is_sentence_transformer = False
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            self.is_sentence_transformer = False
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        
        self.model.eval()
        print(f"Model loaded on: {self.device}")
        print(f"Using model: {self.model_path}")
    
    def get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding using sentence-transformers."""
        embedding = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,  # Important for cosine similarity
            show_progress_bar=False
        )
        return embedding.cpu().numpy().flatten()
    
    def get_bert_embedding(self, text: str, 
                          pooling_strategy: str = 'mean',
                          use_all_layers: bool = False,
                          normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Get BERT embedding with multiple pooling strategies.
        
        Args:
            text: Input text
            pooling_strategy: 'cls', 'mean', 'max', 'weighted', or 'all'
            use_all_layers: Whether to use all hidden layers
            normalize: Whether to L2 normalize the embedding
        
        Returns:
            Dictionary with embedding(s)
        """
        # Tokenize with improved settings
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding='max_length' if self.is_sentence_transformer else True,
            add_special_tokens=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=use_all_layers)
        
        # Get last hidden state
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        embeddings = {}
        
        # CLS pooling (baseline)
        if pooling_strategy in ['cls', 'all']:
            cls_embedding = last_hidden_state[:, 0, :]
            if normalize:
                cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=-1)
            embeddings['cls'] = cls_embedding.cpu().numpy().flatten()
        
        # Mean pooling (often works better than CLS)
        if pooling_strategy in ['mean', 'all']:
            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Sum embeddings along sequence dimension
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            
            mean_pooled = sum_embeddings / sum_mask
            if normalize:
                mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=-1)
            embeddings['mean'] = mean_pooled.cpu().numpy().flatten()
        
        # Max pooling
        if pooling_strategy in ['max', 'all']:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Set padding tokens to -inf so they don't affect max
            masked_hidden = last_hidden_state.clone()
            masked_hidden[input_mask_expanded == 0] = -float('inf')
            
            max_pooled, _ = torch.max(masked_hidden, dim=1)
            if normalize:
                max_pooled = torch.nn.functional.normalize(max_pooled, p=2, dim=-1)
            embeddings['max'] = max_pooled.cpu().numpy().flatten()
        
        # Weighted pooling using attention weights
        if pooling_strategy in ['weighted', 'all'] and hasattr(outputs, 'attentions') and outputs.attentions:
            # Use attention from last layer, average over heads
            last_layer_attention = outputs.attentions[-1].mean(dim=1)  # [batch, seq_len, seq_len]
            # Use CLS token attention to all other tokens
            cls_attention = last_layer_attention[:, 0, :].unsqueeze(-1)  # [batch, seq_len, 1]
            
            weighted_embeddings = torch.sum(last_hidden_state * cls_attention, dim=1)
            if normalize:
                weighted_embeddings = torch.nn.functional.normalize(weighted_embeddings, p=2, dim=-1)
            embeddings['weighted'] = weighted_embeddings.cpu().numpy().flatten()
        
        # Hierarchical pooling (combine last 4 layers)
        if pooling_strategy in ['hierarchical', 'all'] and use_all_layers:
            hierarchical_embeddings = []
            # Use last 4 layers for hierarchical representation
            for layer_idx in range(-4, 0):
                layer_hidden = outputs.hidden_states[layer_idx]
                # Mean pool each layer
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                sum_layer = torch.sum(layer_hidden * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                layer_mean = sum_layer / sum_mask
                hierarchical_embeddings.append(layer_mean)
            
            # Concatenate and reduce
            hierarchical_concat = torch.cat(hierarchical_embeddings, dim=-1)
            # Optional: add a learned projection here if needed
            if normalize:
                hierarchical_concat = torch.nn.functional.normalize(hierarchical_concat, p=2, dim=-1)
            embeddings['hierarchical'] = hierarchical_concat.cpu().numpy().flatten()
        
        return embeddings
    
    def get_embedding(self, text: str, strategy: str = 'mean') -> np.ndarray:
        """Get embedding with specified strategy."""
        if self.is_sentence_transformer:
            return self.get_sentence_transformer_embedding(text)
        else:
            embeddings = self.get_bert_embedding(text, pooling_strategy=strategy)
            return embeddings.get(strategy, embeddings['mean'])

def post_process_embeddings(embeddings: np.ndarray, method: str = 'normalize') -> np.ndarray:
    """
    Post-process embeddings for better quality.
    
    Args:
        embeddings: Raw embeddings
        method: 'normalize', 'whiten', or 'pca'
    
    Returns:
        Processed embeddings
    """
    if method == 'normalize':
        # L2 normalization (crucial for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    elif method == 'whiten':
        # Whitening transformation
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # First standardize
        scaler = StandardScaler()
        standardized = scaler.fit_transform(embeddings)
        
        # Then PCA with whitening
        pca = PCA(whiten=True)
        return pca.fit_transform(standardized)
    
    elif method == 'pca':
        
        from sklearn.decomposition import PCA
        
        n_components = min(768, embeddings.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)
    
    else:
        return embeddings

def save_embeddings(ids: List[str], 
                   embeddings_list: List[np.ndarray], 
                   output_path: str,
                   save_format: str = 'csv'):
    """
    Save embeddings to file.
    
    Args:
        ids: List of molecule IDs
        embeddings_list: List of embedding arrays
        output_path: Output file path
        save_format: 'csv', 'npy', or 'pkl'
    """
    if save_format == 'csv':
        # Convert to comma-separated strings
        embedding_strs = [','.join(map(str, emb.astype(np.float32))) for emb in embeddings_list]
        result = pd.DataFrame({
            'ID': ids,
            'embedding': embedding_strs
        })
        result.to_csv(output_path, index=False)

def main():
    """Main function to generate embeddings for train and validation sets."""
    
    # Configuration
    MODEL_NAME = 'scibert'  # Options: bert_base, scibert, biobert, mpnet, deberta, distilbert
    POOLING_STRATEGY = 'mean'  # Options: cls, mean, max, weighted, hierarchical, all
    POST_PROCESSING = 'normalize'  # Options: none, normalize, whiten, pca
    SAVE_FORMAT = 'csv' 
    
    # Initialize embedding generator
    print("=" * 60)
    print("Enhanced BERT Embedding Generator")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Pooling strategy: {POOLING_STRATEGY}")
    print(f"Post-processing: {POST_PROCESSING}")
    print(f"Save format: {SAVE_FORMAT}")
    print("=" * 60)
    
    embedder = EnhancedBertEmbeddings(model_name=MODEL_NAME, device='auto')
    
    for split in ['train', 'validation']:
        print(f"\n{'='*40}")
        print(f"Processing {split} set...")
        print(f"{'='*40}")
        
        if split == 'train':
            pkl_path = TRAIN_GRAPHS
        else:
            pkl_path = VAL_GRAPHS
        
        print(f"Loading from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        
        ids = []
        all_embeddings = []
        failed_descriptions = 0
        
        for graph in tqdm(graphs, total=len(graphs), desc=f"Generating {split} embeddings"):
            try:
                description = graph.description
                
                if not description or len(description.strip()) == 0:
                    # Handle empty descriptions
                    print(f"Warning: Empty description for graph {graph.id}")
                    # Create zero embedding of appropriate dimension
                    if embedder.is_sentence_transformer:
                        dummy_embedding = np.zeros(768, dtype=np.float32)
                    else:
                        dummy_embedding = np.zeros(embedder.model.config.hidden_size, dtype=np.float32)
                    embeddings = {'mean': dummy_embedding}
                else:
                    # Get embedding
                    if POOLING_STRATEGY == 'all':
                        # Get all strategies
                        embeddings_dict = embedder.get_bert_embedding(
                            description, 
                            pooling_strategy='all',
                            normalize=True
                        )
                        # Use mean pooling as default
                        embeddings = {'mean': embeddings_dict.get('mean', embeddings_dict['cls'])}
                    else:
                        embedding = embedder.get_embedding(description, strategy=POOLING_STRATEGY)
                        embeddings = {'mean': embedding}
                
                ids.append(graph.id)
                all_embeddings.append(embeddings['mean'])
                
            except Exception as e:
                print(f"Error processing graph {graph.id}: {str(e)}")
                failed_descriptions += 1
                # Add zero embedding as fallback
                if embedder.is_sentence_transformer:
                    dummy_embedding = np.zeros(768, dtype=np.float32)
                else:
                    dummy_embedding = np.zeros(embedder.model.config.hidden_size, dtype=np.float32)
                ids.append(graph.id)
                all_embeddings.append(dummy_embedding)
        
        if failed_descriptions > 0:
            print(f"Warning: Failed to process {failed_descriptions} descriptions")
        
        # Post-process embeddings
        print("Post-processing embeddings...")
        embeddings_array = np.stack(all_embeddings)
        
        if POST_PROCESSING != 'none':
            embeddings_array = post_process_embeddings(embeddings_array, method=POST_PROCESSING)
        
        # Create output filename with configuration
        config_str = f"{MODEL_NAME}_{POOLING_STRATEGY}_{POST_PROCESSING}"
        if SAVE_FORMAT == 'csv':
            output_path = os.path.join(BASE, f'{split}_embeddings_{config_str}.csv')
        elif SAVE_FORMAT == 'npy':
            output_path = os.path.join(BASE, f'{split}_embeddings_{config_str}.npz')
        else:
            output_path = os.path.join(BASE, f'{split}_embeddings_{config_str}.pkl')
        
        # Save embeddings
        save_embeddings(ids, list(embeddings_array), output_path, save_format=SAVE_FORMAT)
        
        # Print embedding statistics
        print(f"\nEmbedding statistics for {split}:")
        print(f"  Shape: {embeddings_array.shape}")
        print(f"  Mean norm: {np.mean(np.linalg.norm(embeddings_array, axis=1)):.4f}")
        print(f"  Std norm: {np.std(np.linalg.norm(embeddings_array, axis=1)):.4f}")
        
        # Save additional metadata
        metadata = {
            'split': split,
            'model': MODEL_NAME,
            'pooling_strategy': POOLING_STRATEGY,
            'post_processing': POST_PROCESSING,
            'embedding_dim': embeddings_array.shape[1],
            'num_samples': len(ids),
            'failed_descriptions': failed_descriptions
        }
        
        metadata_path = os.path.join(BASE, f'{split}_embeddings_metadata_{config_str}.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

def benchmark_strategies():
    """Benchmark different embedding strategies on a subset."""
    print("\n" + "="*60)
    print("Benchmarking different embedding strategies")
    print("="*60)
    
    with open(TRAIN_GRAPHS, 'rb') as f:
        graphs = pickle.load(f)
    
    test_graphs = graphs[:100]  # Use first 100 for benchmarking
    
    strategies = ['cls', 'mean', 'max', 'weighted']
    models_to_test = ['bert_base', 'mpnet', 'scibert']
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        embedder = EnhancedBertEmbeddings(model_name=model_name, device='auto')
        
        for strategy in strategies:
            print(f"  Strategy: {strategy}")
            
            embeddings = []
            times = []
            
            for graph in test_graphs[:10]:  
                start_time = time.time()
                
                try:
                    if embedder.is_sentence_transformer and strategy != 'cls':
                        # Sentence transformers use their own pooling
                        embedding = embedder.get_embedding(graph.description, strategy='mean')
                    else:
                        embedding = embedder.get_embedding(graph.description, strategy=strategy)
                    embeddings.append(embedding)
                except:
                    embeddings.append(np.zeros(768))
                
                times.append(time.time() - start_time)
            
            if embeddings:
                avg_time = np.mean(times)
                avg_norm = np.mean([np.linalg.norm(emb) for emb in embeddings])
                results[f"{model_name}_{strategy}"] = {
                    'avg_time': avg_time,
                    'avg_norm': avg_norm,
                    'dimension': len(embeddings[0])
                }
                print(f"    Avg time: {avg_time:.4f}s, Avg norm: {avg_norm:.4f}, Dim: {len(embeddings[0])}")
    
    # Save benchmark results
    benchmark_path = os.path.join(BASE, 'embedding_benchmark_results.json')
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nBenchmark results saved to: {benchmark_path}")
    return results

if __name__ == "__main__":
    benchmark_strategies()
    
    main()
    
    print("\n" + "="*60)
    print("Embedding generation complete!")
    print("="*60)