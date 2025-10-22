#!/usr/bin/env python3
"""
Embedding Analysis Module

Handles geometric analysis of V-JEPA2 embeddings including trajectory analysis,
clustering, and distance metrics.
"""

from typing import Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingAnalyzer:
    """Analyzes geometric properties of V-JEPA2 embeddings."""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize embedding analyzer.
        
        Args:
            n_clusters: Number of clusters for K-means
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def analyze_trajectory(self, embeddings: torch.Tensor, 
                          events: Optional[Dict[str, Tuple[int, int]]] = None,
                          auto_events: Optional[List[int]] = None) -> Dict:
        """
        Analyze the geometric properties of embeddings.
        
        Args:
            embeddings: Embeddings tensor [batch_size, num_patches, embedding_dim]
            events: Manual event annotations
            auto_events: Automatically detected events
            
        Returns:
            Dictionary containing analysis results
        """
        print("Analyzing embedding trajectory...")
        
        # Normalize input to shape [num_samples, embedding_dim]
        emb = embeddings.detach().cpu()
        if emb.ndim == 3:
            b, s, d = emb.shape
            if b > 1:
                # Average over sequence tokens per sample
                embeddings_flat = einops.reduce(emb.numpy(), 'b s d -> b d', reduction='mean')
            else:
                # Treat sequence tokens as samples when batch==1
                embeddings_flat = emb.squeeze(0).numpy()  # [s, d]
        elif emb.ndim == 2:
            embeddings_flat = emb.numpy()
        else:
            # Fallback: collapse all but last dim into samples
            last_dim = emb.shape[-1]
            embeddings_flat = emb.reshape(-1, last_dim).numpy()
        
        # Compute distances between consecutive frames
        distances = np.linalg.norm(np.diff(embeddings_flat, axis=0), axis=1)
        
        # Compute cosine similarities
        similarities = []
        for i in range(len(embeddings_flat) - 1):
            sim = cosine_similarity(
                embeddings_flat[i:i+1], 
                embeddings_flat[i+1:i+2]
            )[0, 0]
            similarities.append(sim)
        
        # Apply PCA for visualization (robust to small sample counts)
        n_samples, n_features = embeddings_flat.shape
        n_components = 2 if min(n_samples, n_features) >= 2 else 1
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(embeddings_flat)
        # Pad to 2D for downstream plotting
        if n_components == 1:
            embeddings_pca = np.concatenate([embeddings_pca, np.zeros_like(embeddings_pca)], axis=1)
        
        # K-means clustering
        n_clusters = min(self.n_clusters, max(1, len(embeddings_flat) // 10))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(embeddings_flat)
        else:
            cluster_labels = np.zeros(len(embeddings_flat))
        
        # Event impact analysis
        event_stats = {}
        if events:
            event_stats = self._analyze_event_impact(embeddings_flat, events)
        
        return {
            'embeddings_pca': embeddings_pca,
            'embeddings_flat': embeddings_flat,
            'distances': distances,
            'similarities': similarities,
            'cluster_labels': cluster_labels,
            'event_stats': event_stats,
            'auto_events': auto_events,
            'explained_variance': pca.explained_variance_ratio_ if hasattr(pca, 'explained_variance_ratio_') else np.array([1.0, 0.0]),
            'pca_components': pca.components_,
            'cluster_centers': kmeans.cluster_centers_ if n_clusters > 1 else None
        }
    
    def _analyze_event_impact(self, embeddings_flat: np.ndarray, 
                             events: Dict[str, Tuple[int, int]]) -> Dict:
        """
        Analyze the impact of events on embeddings.
        
        Args:
            embeddings_flat: Flattened embeddings
            events: Event annotations
            
        Returns:
            Dictionary with event impact statistics
        """
        event_stats = {}
        
        for event_name, (start, end) in events.items():
            if start < len(embeddings_flat) and end < len(embeddings_flat):
                event_embeddings = embeddings_flat[start:end+1]
                non_event_embeddings = np.vstack([
                    embeddings_flat[:start],
                    embeddings_flat[end+1:]
                ])
                
                if len(non_event_embeddings) > 0:
                    # Compute mean distances
                    event_mean = np.mean(np.linalg.norm(event_embeddings, axis=1))
                    non_event_mean = np.mean(np.linalg.norm(non_event_embeddings, axis=1))
                    
                    # Compute pairwise distances within event
                    event_pairwise = self._compute_pairwise_distances(event_embeddings)
                    non_event_pairwise = self._compute_pairwise_distances(non_event_embeddings)
                    
                    event_stats[event_name] = {
                        'event_magnitude': event_mean,
                        'non_event_magnitude': non_event_mean,
                        'magnitude_ratio': event_mean / non_event_mean if non_event_mean > 0 else 0,
                        'event_pairwise_mean': np.mean(event_pairwise),
                        'non_event_pairwise_mean': np.mean(non_event_pairwise),
                        'frames': (start, end),
                        'num_frames': end - start + 1
                    }
        
        return event_stats
    
    def _compute_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between embeddings."""
        if len(embeddings) < 2:
            return np.array([])
        
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        return np.array(distances)
    
    def compute_embedding_statistics(self, embeddings: torch.Tensor) -> Dict:
        """
        Compute basic statistics of embeddings.
        
        Args:
            embeddings: Embeddings tensor
            
        Returns:
            Dictionary with embedding statistics
        """
        embeddings_cpu = embeddings.cpu().numpy()
        
        stats = {
            'shape': embeddings_cpu.shape,
            'mean': np.mean(embeddings_cpu),
            'std': np.std(embeddings_cpu),
            'min': np.min(embeddings_cpu),
            'max': np.max(embeddings_cpu),
            'norm_mean': np.mean(np.linalg.norm(embeddings_cpu, axis=-1)),
            'norm_std': np.std(np.linalg.norm(embeddings_cpu, axis=-1))
        }
        
        return stats
    
    def analyze_temporal_dynamics(self, embeddings: torch.Tensor) -> Dict:
        """
        Analyze temporal dynamics of embeddings.
        
        Args:
            embeddings: Embeddings tensor [batch_size, num_patches, embedding_dim]
            
        Returns:
            Dictionary with temporal analysis results
        """
        embeddings_cpu = embeddings.cpu().numpy()
        
        # Handle different input shapes
        if embeddings_cpu.ndim == 3:
            # [batch, seq, dim] - treat sequence as temporal dimension
            if embeddings_cpu.shape[0] == 1:
                # Single batch: use sequence dimension
                embeddings_temporal = embeddings_cpu[0]  # [seq, dim]
            else:
                # Multiple batches: average over batch dimension
                embeddings_temporal = embeddings_cpu.mean(axis=0)  # [seq, dim]
        elif embeddings_cpu.ndim == 2:
            # [seq, dim] - already temporal
            embeddings_temporal = embeddings_cpu
        else:
            # Flatten to 2D
            embeddings_temporal = embeddings_cpu.reshape(-1, embeddings_cpu.shape[-1])
        
        # Ensure we have enough temporal samples
        if len(embeddings_temporal) < 2:
            return {
                'velocity_magnitude': np.array([]),
                'acceleration_magnitude': np.array([]),
                'velocity_peaks': np.array([]),
                'acceleration_peaks': np.array([]),
                'mean_velocity': 0.0,
                'mean_acceleration': 0.0
            }
        
        # Compute temporal derivatives
        temporal_diff = np.diff(embeddings_temporal, axis=0)
        
        # Compute velocity magnitude (ensure 1D)
        velocity_magnitude = np.linalg.norm(temporal_diff, axis=-1)
        if velocity_magnitude.ndim > 1:
            velocity_magnitude = velocity_magnitude.flatten()
        
        # Compute acceleration if we have enough samples
        if len(temporal_diff) > 1:
            temporal_acceleration = np.diff(temporal_diff, axis=0)
            acceleration_magnitude = np.linalg.norm(temporal_acceleration, axis=-1)
            if acceleration_magnitude.ndim > 1:
                acceleration_magnitude = acceleration_magnitude.flatten()
        else:
            acceleration_magnitude = np.array([])
        
        # Find peaks in velocity and acceleration
        from scipy.signal import find_peaks
        
        velocity_peaks = np.array([])
        acceleration_peaks = np.array([])
        
        if len(velocity_magnitude) > 0 and np.std(velocity_magnitude) > 0:
            try:
                velocity_peaks, _ = find_peaks(velocity_magnitude, height=np.mean(velocity_magnitude))
            except:
                velocity_peaks = np.array([])
        
        if len(acceleration_magnitude) > 0 and np.std(acceleration_magnitude) > 0:
            try:
                acceleration_peaks, _ = find_peaks(acceleration_magnitude, height=np.mean(acceleration_magnitude))
            except:
                acceleration_peaks = np.array([])
        
        return {
            'velocity_magnitude': velocity_magnitude,
            'acceleration_magnitude': acceleration_magnitude,
            'velocity_peaks': velocity_peaks,
            'acceleration_peaks': acceleration_peaks,
            'mean_velocity': np.mean(velocity_magnitude) if len(velocity_magnitude) > 0 else 0.0,
            'mean_acceleration': np.mean(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 0.0
        }
    
    def compute_embedding_similarity_matrix(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Compute pairwise similarity matrix between embeddings.
        
        Args:
            embeddings: Embeddings tensor [batch_size, num_patches, embedding_dim]
            
        Returns:
            Similarity matrix
        """
        embeddings_cpu = embeddings.cpu().numpy()
        
        # Flatten spatial dimensions
        embeddings_flat = einops.reduce(embeddings_cpu, 'b p d -> b d', reduction='mean')
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings_flat)
        
        return similarity_matrix
    
    def detect_embedding_anomalies(self, embeddings: torch.Tensor, 
                                  threshold: float = 2.0) -> List[int]:
        """
        Detect anomalous embeddings using statistical methods.
        
        Args:
            embeddings: Embeddings tensor
            threshold: Threshold for anomaly detection (in standard deviations)
            
        Returns:
            List of frame indices with anomalous embeddings
        """
        embeddings_cpu = embeddings.cpu().numpy()
        
        # Flatten spatial dimensions
        embeddings_flat = einops.reduce(embeddings_cpu, 'b p d -> b d', reduction='mean')
        
        # Compute embedding norms
        norms = np.linalg.norm(embeddings_flat, axis=1)
        
        # Detect outliers using z-score
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        z_scores = np.abs((norms - mean_norm) / std_norm)
        anomalies = np.where(z_scores > threshold)[0].tolist()
        
        print(f"Detected {len(anomalies)} anomalous embeddings at frames: {anomalies}")
        
        return anomalies
    
    def export_analysis_results(self, analysis_results: Dict, 
                               output_path: str) -> None:
        """
        Export analysis results to files.
        
        Args:
            analysis_results: Results from trajectory analysis
            output_path: Base path for output files
        """
        import json
        import numpy as np
        
        # Prepare results for JSON serialization
        results_json = {
            'explained_variance': analysis_results['explained_variance'].tolist(),
            'event_stats': analysis_results['event_stats'],
            'auto_events': analysis_results['auto_events'],
            'distance_stats': {
                'mean': float(np.mean(analysis_results['distances'])),
                'std': float(np.std(analysis_results['distances'])),
                'min': float(np.min(analysis_results['distances'])),
                'max': float(np.max(analysis_results['distances']))
            },
            'similarity_stats': {
                'mean': float(np.mean(analysis_results['similarities'])),
                'std': float(np.std(analysis_results['similarities'])),
                'min': float(np.min(analysis_results['similarities'])),
                'max': float(np.max(analysis_results['similarities']))
            }
        }
        
        # Save JSON results
        with open(f"{output_path}_analysis.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save numpy arrays
        np.savez(f"{output_path}_data.npz",
                 embeddings_pca=analysis_results['embeddings_pca'],
                 embeddings_flat=analysis_results['embeddings_flat'],
                 distances=analysis_results['distances'],
                 similarities=analysis_results['similarities'],
                 cluster_labels=analysis_results['cluster_labels'])
        
        print(f"Analysis results exported to {output_path}_*")


def main():
    """Test the embedding analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test embedding analyzer")
    parser.add_argument("--embeddings", help="Path to embeddings .npz file")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = EmbeddingAnalyzer()
    
    if args.embeddings:
        # Load embeddings from file
        data = np.load(args.embeddings)
        embeddings = torch.from_numpy(data['embeddings'])
    else:
        # Create dummy embeddings for testing
        embeddings = torch.randn(64, 1024, 1024)  # [frames, patches, dim]
    
    # Analyze trajectory
    results = analyzer.analyze_trajectory(embeddings)
    
    # Compute statistics
    stats = analyzer.compute_embedding_statistics(embeddings)
    print("Embedding Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Analyze temporal dynamics
    temporal_results = analyzer.analyze_temporal_dynamics(embeddings)
    print(f"Temporal Analysis: {len(temporal_results['velocity_peaks'])} velocity peaks")
    
    # Detect anomalies
    anomalies = analyzer.detect_embedding_anomalies(embeddings)
    
    # Export results
    os.makedirs(args.output_dir, exist_ok=True)
    analyzer.export_analysis_results(results, f"{args.output_dir}/test_analysis")
    
    print("Embedding analysis test complete!")


if __name__ == "__main__":
    main()
