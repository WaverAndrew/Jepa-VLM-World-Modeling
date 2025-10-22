#!/usr/bin/env python3
"""
Sliding Window Analysis Module

Implements sliding window analysis to compare consecutive video clips and detect
scene changes through cosine similarity analysis.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


class SlidingWindowAnalyzer:
    """Analyzes video sequences using sliding window approach."""
    
    def __init__(self, window_size: int = 16, stride: int = 8, 
                 overlap_threshold: float = 0.7):
        """
        Initialize sliding window analyzer.
        
        Args:
            window_size: Number of frames per window
            stride: Step size between windows
            overlap_threshold: Threshold for detecting significant changes
        """
        self.window_size = window_size
        self.stride = stride
        self.overlap_threshold = overlap_threshold
    
    def create_sliding_windows(self, frames: np.ndarray) -> List[np.ndarray]:
        """
        Create sliding windows from video frames.
        
        Args:
            frames: Array of video frames [num_frames, height, width, channels]
            
        Returns:
            List of window arrays
        """
        num_frames = len(frames)
        windows = []
        
        for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = frames[start_idx:end_idx]
            windows.append(window)
        
        print(f"Created {len(windows)} sliding windows (size={self.window_size}, stride={self.stride})")
        return windows
    
    def encode_windows(self, windows: List[np.ndarray], 
                      model_wrapper, video_processor) -> List[torch.Tensor]:
        """
        Encode sliding windows using V-JEPA2 model.
        
        Args:
            windows: List of window arrays
            model_wrapper: V-JEPA2 model wrapper
            video_processor: Video processor
            
        Returns:
            List of embedding tensors for each window
        """
        print("Encoding sliding windows...")
        embeddings = []
        
        for i, window in enumerate(windows):
            # Preprocess window
            window_tensor = video_processor.preprocess_frames(window)
            
            # Get model inputs
            model_inputs = model_wrapper.preprocess_video(window_tensor)
            
            # Extract embeddings
            window_embeddings = model_wrapper.extract_embeddings(model_inputs)
            
            # Average over spatial dimensions to get window-level embedding
            if window_embeddings.dim() == 3:  # [batch, seq, dim]
                window_embedding = window_embeddings.mean(dim=1)  # [batch, dim]
            else:
                window_embedding = window_embeddings
            
            embeddings.append(window_embedding)
            
            if (i + 1) % 10 == 0:
                print(f"Encoded {i + 1}/{len(windows)} windows")
        
        print(f"Encoded {len(embeddings)} windows")
        return embeddings
    
    def compute_window_similarities(self, embeddings: List[torch.Tensor]) -> np.ndarray:
        """
        Compute cosine similarities between consecutive windows.
        
        Args:
            embeddings: List of window embeddings
            
        Returns:
            Array of cosine similarities between consecutive windows
        """
        print("Computing window similarities...")
        
        similarities = []
        
        for i in range(len(embeddings) - 1):
            # Get embeddings for current and next window
            curr_emb = embeddings[i].cpu().numpy()
            next_emb = embeddings[i + 1].cpu().numpy()
            
            # Compute cosine similarity
            sim = cosine_similarity(curr_emb, next_emb)[0, 0]
            similarities.append(sim)
        
        similarities = np.array(similarities)
        print(f"Computed {len(similarities)} window similarities")
        return similarities
    
    def detect_scene_changes(self, similarities: np.ndarray, 
                           threshold: Optional[float] = None) -> List[int]:
        """
        Detect scene changes based on similarity drops.
        
        Args:
            similarities: Array of cosine similarities
            threshold: Threshold for scene change detection
            
        Returns:
            List of window indices where scene changes are detected
        """
        if threshold is None:
            threshold = 1.0 - self.overlap_threshold
        
        # Find windows where similarity drops significantly
        scene_changes = []
        
        for i, sim in enumerate(similarities):
            if sim < threshold:
                scene_changes.append(i)
        
        print(f"Detected {len(scene_changes)} scene changes at windows: {scene_changes}")
        return scene_changes
    
    def analyze_temporal_dynamics(self, similarities: np.ndarray) -> Dict:
        """
        Analyze temporal dynamics of window similarities.
        
        Args:
            similarities: Array of cosine similarities
            
        Returns:
            Dictionary with temporal analysis results
        """
        # Compute derivatives
        first_derivative = np.gradient(similarities)
        second_derivative = np.gradient(first_derivative)
        
        # Find peaks and valleys
        from scipy.signal import find_peaks
        
        # Peaks (high similarity)
        peaks, _ = find_peaks(similarities, height=np.mean(similarities))
        
        # Valleys (low similarity - potential scene changes)
        valleys, _ = find_peaks(-similarities, height=-np.mean(similarities))
        
        # Find rapid changes (high absolute derivative)
        rapid_changes = np.where(np.abs(first_derivative) > np.std(first_derivative) * 2)[0]
        
        return {
            'similarities': similarities,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative,
            'peaks': peaks,
            'valleys': valleys,
            'rapid_changes': rapid_changes,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def create_similarity_plot(self, similarities: np.ndarray, 
                             scene_changes: List[int], 
                             temporal_analysis: Dict,
                             video_name: str, output_dir: str) -> str:
        """
        Create sliding window similarity plot.
        
        Args:
            similarities: Array of cosine similarities
            scene_changes: List of detected scene changes
            temporal_analysis: Temporal analysis results
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating sliding window similarity plot...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Main similarity plot
        window_indices = np.arange(len(similarities))
        ax1.plot(window_indices, similarities, 'b-', linewidth=2, label='Cosine Similarity')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title(f'Sliding Window Similarity Analysis - {video_name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight scene changes
        if scene_changes:
            ax1.scatter(scene_changes, similarities[scene_changes], 
                       c='red', s=100, marker='*', 
                       label='Detected Scene Changes', zorder=5)
            ax1.legend()
        
        # Add threshold line
        threshold = 1.0 - self.overlap_threshold
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Change Threshold ({threshold:.2f})')
        ax1.legend()
        
        # First derivative plot
        ax2.plot(window_indices, temporal_analysis['first_derivative'], 
                'g-', linewidth=2, label='First Derivative')
        ax2.set_ylabel('Similarity Change Rate')
        ax2.set_title('Temporal Dynamics - First Derivative')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight rapid changes
        rapid_changes = temporal_analysis['rapid_changes']
        if len(rapid_changes) > 0:
            ax2.scatter(rapid_changes, temporal_analysis['first_derivative'][rapid_changes],
                       c='orange', s=50, marker='o', 
                       label='Rapid Changes', zorder=5)
            ax2.legend()
        
        # Second derivative plot
        ax3.plot(window_indices, temporal_analysis['second_derivative'], 
                'purple', linewidth=2, label='Second Derivative')
        ax3.set_xlabel('Window Index')
        ax3.set_ylabel('Acceleration of Change')
        ax3.set_title('Temporal Dynamics - Second Derivative')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/{video_name}_sliding_window_similarity.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_window_timeline(self, similarities: np.ndarray, 
                             scene_changes: List[int],
                             video_name: str, output_dir: str) -> str:
        """
        Create a timeline visualization showing window positions and similarities.
        
        Args:
            similarities: Array of cosine similarities
            scene_changes: List of detected scene changes
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating window timeline...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create timeline
        window_indices = np.arange(len(similarities))
        
        # Color-code by similarity
        colors = plt.cm.viridis(similarities)
        
        # Plot timeline bars
        for i, sim in enumerate(similarities):
            start_frame = i * self.stride
            end_frame = start_frame + self.window_size
            
            ax.barh(0, self.window_size, left=start_frame, height=0.8, 
                   color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add similarity text
            ax.text(start_frame + self.window_size/2, 0, f'{sim:.2f}', 
                   ha='center', va='center', fontsize=8, rotation=90)
        
        # Highlight scene changes
        for change_idx in scene_changes:
            start_frame = change_idx * self.stride
            ax.axvline(x=start_frame, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Window')
        ax.set_title(f'Sliding Window Timeline - {video_name}')
        ax.set_ylim(-0.5, 0.5)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                 norm=plt.Normalize(vmin=similarities.min(), vmax=similarities.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Cosine Similarity')
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/{video_name}_window_timeline.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def export_results(self, similarities: np.ndarray, 
                      scene_changes: List[int],
                      temporal_analysis: Dict,
                      video_name: str, output_dir: str) -> None:
        """
        Export sliding window analysis results.
        
        Args:
            similarities: Array of cosine similarities
            scene_changes: List of detected scene changes
            temporal_analysis: Temporal analysis results
            video_name: Name of the video
            output_dir: Output directory
        """
        print("Exporting sliding window results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export CSV data
        import pandas as pd
        
        window_data = []
        for i, sim in enumerate(similarities):
            start_frame = i * self.stride
            end_frame = start_frame + self.window_size
            
            window_data.append({
                'window_index': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'cosine_similarity': sim,
                'is_scene_change': i in scene_changes,
                'first_derivative': temporal_analysis['first_derivative'][i],
                'second_derivative': temporal_analysis['second_derivative'][i]
            })
        
        df = pd.DataFrame(window_data)
        df.to_csv(f"{output_dir}/{video_name}_sliding_window_data.csv", index=False)
        
        # Export JSON summary
        import json
        
        summary = {
            'video_name': video_name,
            'window_size': self.window_size,
            'stride': self.stride,
            'overlap_threshold': self.overlap_threshold,
            'num_windows': len(similarities),
            'num_scene_changes': len(scene_changes),
            'scene_change_windows': scene_changes,
            'similarity_stats': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            },
            'temporal_stats': {
                'num_peaks': len(temporal_analysis['peaks']),
                'num_valleys': len(temporal_analysis['valleys']),
                'num_rapid_changes': len(temporal_analysis['rapid_changes'])
            }
        }
        
        with open(f"{output_dir}/{video_name}_sliding_window_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Sliding window results exported to {output_dir}/")


def main():
    """Test the sliding window analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sliding window analyzer")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    parser.add_argument("--window-size", type=int, default=16, help="Window size")
    parser.add_argument("--stride", type=int, default=8, help="Stride between windows")
    parser.add_argument("--overlap-threshold", type=float, default=0.7, 
                       help="Overlap threshold for scene change detection")
    
    args = parser.parse_args()
    
    # Import other modules
    from video_processor import VideoProcessor
    from vjepa_model import VJEPA2Model
    
    # Initialize components
    analyzer = SlidingWindowAnalyzer(
        window_size=args.window_size,
        stride=args.stride,
        overlap_threshold=args.overlap_threshold
    )
    
    processor = VideoProcessor()
    model = VJEPA2Model()
    
    # Load video
    frames = processor.load_video(args.video, num_frames=128)  # More frames for sliding window
    
    # Create sliding windows
    windows = analyzer.create_sliding_windows(frames)
    
    # Encode windows
    embeddings = analyzer.encode_windows(windows, model, processor)
    
    # Compute similarities
    similarities = analyzer.compute_window_similarities(embeddings)
    
    # Detect scene changes
    scene_changes = analyzer.detect_scene_changes(similarities)
    
    # Analyze temporal dynamics
    temporal_analysis = analyzer.analyze_temporal_dynamics(similarities)
    
    # Create visualizations
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    
    similarity_plot = analyzer.create_similarity_plot(
        similarities, scene_changes, temporal_analysis, video_name, args.output_dir
    )
    
    timeline_plot = analyzer.create_window_timeline(
        similarities, scene_changes, video_name, args.output_dir
    )
    
    # Export results
    analyzer.export_results(
        similarities, scene_changes, temporal_analysis, video_name, args.output_dir
    )
    
    print(f"Sliding window analysis complete!")
    print(f"Similarity plot: {similarity_plot}")
    print(f"Timeline plot: {timeline_plot}")


if __name__ == "__main__":
    main()
