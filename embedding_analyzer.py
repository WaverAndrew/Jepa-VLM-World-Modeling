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
        
        # Move to CPU for analysis
        embeddings_cpu = embeddings.cpu().numpy()
        
        # Flatten spatial dimensions for trajectory analysis
        # Shape: (batch_size, num_patches, embedding_dim) -> (batch_size, embedding_dim)
        embeddings_flat = einops.reduce(embeddings_cpu, 'b p d -> b d', reduction='mean')
        
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
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_flat)
        
        # K-means clustering
        n_clusters = min(self.n_clusters, len(embeddings_flat) // 10)
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
            'explained_variance': pca.explained_variance_ratio_,
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
        
        # Compute temporal derivatives
        temporal_diff = np.diff(embeddings_cpu, axis=0)
        temporal_acceleration = np.diff(temporal_diff, axis=0)
        
        # Compute velocity and acceleration magnitudes
        velocity_magnitude = np.linalg.norm(temporal_diff, axis=-1)
        acceleration_magnitude = np.linalg.norm(temporal_acceleration, axis=-1)
        
        # Find peaks in velocity and acceleration
        from scipy.signal import find_peaks
        
        velocity_peaks, _ = find_peaks(velocity_magnitude, height=np.mean(velocity_magnitude))
        acceleration_peaks, _ = find_peaks(acceleration_magnitude, height=np.mean(acceleration_magnitude))
        
        return {
            'velocity_magnitude': velocity_magnitude,
            'acceleration_magnitude': acceleration_magnitude,
            'velocity_peaks': velocity_peaks,
            'acceleration_peaks': acceleration_peaks,
            'mean_velocity': np.mean(velocity_magnitude),
            'mean_acceleration': np.mean(acceleration_magnitude)
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
