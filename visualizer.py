#!/usr/bin/env python3
"""
Visualization Module

Creates publication-ready visualizations for V-JEPA2 embedding analysis.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle


class EmbeddingVisualizer:
    """Creates visualizations for embedding analysis results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_trajectory(self, analysis_results: Dict, video_name: str, 
                       output_dir: str, show_3d: bool = False) -> str:
        """
        Create embedding trajectory plot.
        
        Args:
            analysis_results: Results from trajectory analysis
            video_name: Name of the video
            output_dir: Output directory
            show_3d: Whether to create 3D plot
            
        Returns:
            Path to saved plot
        """
        print("Creating trajectory plot...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_pca = analysis_results['embeddings_pca']
        cluster_labels = analysis_results['cluster_labels']
        auto_events = analysis_results.get('auto_events', [])
        explained_variance = analysis_results['explained_variance']
        
        if show_3d:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 3D trajectory plot
            ax.plot(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                   embeddings_pca[:, 2], 'o-', alpha=0.7, linewidth=2, markersize=6)
            
            # Color by cluster
            scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                               embeddings_pca[:, 2], c=cluster_labels, cmap='tab10', s=100, alpha=0.8)
            
            # Highlight events
            if auto_events:
                ax.scatter(embeddings_pca[auto_events, 0], 
                          embeddings_pca[auto_events, 1],
                          embeddings_pca[auto_events, 2],
                          c='red', s=200, marker='*', 
                          label='Auto-detected Events', zorder=5)
            
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
            ax.set_zlabel(f"PC3 ({explained_variance[2]:.1%} variance)")
            ax.set_title(f"V-JEPA2 Embedding Trajectory (3D) - {video_name}")
            
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # 2D trajectory plot
            ax.plot(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                   'o-', alpha=0.7, linewidth=2, markersize=6)
            
            # Color by cluster
            scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                               c=cluster_labels, cmap='tab10', s=100, alpha=0.8)
            
            # Highlight events
            if auto_events:
                ax.scatter(embeddings_pca[auto_events, 0], 
                          embeddings_pca[auto_events, 1],
                          c='red', s=200, marker='*', 
                          label='Auto-detected Events', zorder=5)
            
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
            ax.set_title(f"V-JEPA2 Embedding Trajectory - {video_name}")
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        suffix = "_3d" if show_3d else ""
        plot_path = f"{output_dir}/{video_name}_trajectory{suffix}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_distance_analysis(self, analysis_results: Dict, video_name: str, 
                             output_dir: str) -> str:
        """
        Create distance and similarity analysis plots.
        
        Args:
            analysis_results: Results from trajectory analysis
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating distance analysis plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        distances = analysis_results['distances']
        similarities = analysis_results['similarities']
        auto_events = analysis_results.get('auto_events', [])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Distance plot
        ax1.plot(distances, linewidth=2, color='blue', label='Euclidean Distance')
        ax1.set_ylabel('Distance')
        ax1.set_title('Inter-frame Embedding Distances')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight events
        if auto_events:
            for event_frame in auto_events:
                if event_frame < len(distances):
                    ax1.axvline(x=event_frame, color='red', linestyle='--', alpha=0.7)
        
        # Similarity plot
        ax2.plot(similarities, linewidth=2, color='green', label='Cosine Similarity')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Similarity')
        ax2.set_title('Inter-frame Embedding Similarities')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight events
        if auto_events:
            for event_frame in auto_events:
                if event_frame < len(similarities):
                    ax2.axvline(x=event_frame, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/{video_name}_distances.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_cluster_analysis(self, analysis_results: Dict, video_name: str, 
                            output_dir: str) -> str:
        """
        Create cluster analysis plots.
        
        Args:
            analysis_results: Results from trajectory analysis
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating cluster analysis plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cluster_labels = analysis_results['cluster_labels']
        auto_events = analysis_results.get('auto_events', [])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Cluster assignment over time
        ax1.plot(cluster_labels, linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Cluster ID')
        ax1.set_title('Cluster Assignment Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Highlight events
        if auto_events:
            for event_frame in auto_events:
                ax1.axvline(x=event_frame, color='red', linestyle='--', alpha=0.7)
        
        # Cluster distribution
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        ax2.bar(unique_clusters, counts, alpha=0.7)
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Frames')
        ax2.set_title('Cluster Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/{video_name}_clusters.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_event_impact(self, analysis_results: Dict, video_name: str, 
                         output_dir: str) -> str:
        """
        Create event impact analysis plots.
        
        Args:
            analysis_results: Results from trajectory analysis
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating event impact plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        event_stats = analysis_results.get('event_stats', {})
        
        if not event_stats:
            print("No event statistics available for plotting")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Event magnitude comparison
        event_names = list(event_stats.keys())
        event_magnitudes = [stats['event_magnitude'] for stats in event_stats.values()]
        non_event_magnitudes = [stats['non_event_magnitude'] for stats in event_stats.values()]
        
        x = np.arange(len(event_names))
        width = 0.35
        
        ax1.bar(x - width/2, event_magnitudes, width, label='Event Frames', alpha=0.7)
        ax1.bar(x + width/2, non_event_magnitudes, width, label='Non-Event Frames', alpha=0.7)
        
        ax1.set_xlabel('Events')
        ax1.set_ylabel('Embedding Magnitude')
        ax1.set_title('Event vs Non-Event Embedding Magnitudes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(event_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Magnitude ratios
        magnitude_ratios = [stats['magnitude_ratio'] for stats in event_stats.values()]
        
        ax2.bar(event_names, magnitude_ratios, alpha=0.7, color='orange')
        ax2.set_xlabel('Events')
        ax2.set_ylabel('Magnitude Ratio (Event/Non-Event)')
        ax2.set_title('Event Impact Magnitude Ratios')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/{video_name}_event_impact.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_heatmap(self, similarity_matrix: np.ndarray, video_name: str, 
                    output_dir: str) -> str:
        """
        Create similarity heatmap.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating similarity heatmap...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Frame Index')
        ax.set_title(f'Embedding Similarity Heatmap - {video_name}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity')
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/{video_name}_similarity_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_summary_plot(self, analysis_results: Dict, video_name: str, 
                          output_dir: str) -> str:
        """
        Create a comprehensive summary plot.
        
        Args:
            analysis_results: Results from trajectory analysis
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            Path to saved plot
        """
        print("Creating summary plot...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_pca = analysis_results['embeddings_pca']
        distances = analysis_results['distances']
        similarities = analysis_results['similarities']
        cluster_labels = analysis_results['cluster_labels']
        auto_events = analysis_results.get('auto_events', [])
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Trajectory plot
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(embeddings_pca[:, 0], embeddings_pca[:, 1], 'o-', alpha=0.7, linewidth=2)
        scatter = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                           c=cluster_labels, cmap='tab10', s=100, alpha=0.8)
        if auto_events:
            ax1.scatter(embeddings_pca[auto_events, 0], 
                      embeddings_pca[auto_events, 1],
                      c='red', s=200, marker='*', zorder=5)
        ax1.set_title('Embedding Trajectory')
        ax1.grid(True, alpha=0.3)
        
        # Distance plot
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(distances, linewidth=2, color='blue')
        if auto_events:
            for event_frame in auto_events:
                if event_frame < len(distances):
                    ax2.axvline(x=event_frame, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Inter-frame Distances')
        ax2.grid(True, alpha=0.3)
        
        # Similarity plot
        ax3 = fig.add_subplot(gs[2, :2])
        ax3.plot(similarities, linewidth=2, color='green')
        if auto_events:
            for event_frame in auto_events:
                if event_frame < len(similarities):
                    ax3.axvline(x=event_frame, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Inter-frame Similarities')
        ax3.set_xlabel('Frame Index')
        ax3.grid(True, alpha=0.3)
        
        # Cluster timeline
        ax4 = fig.add_subplot(gs[:, 2])
        ax4.plot(cluster_labels, linewidth=2, marker='o', markersize=4)
        if auto_events:
            for event_frame in auto_events:
                ax4.axvline(x=event_frame, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Cluster Timeline')
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel('Cluster ID')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'V-JEPA2 Analysis Summary - {video_name}', fontsize=16)
        
        plot_path = f"{output_dir}/{video_name}_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_all_visualizations(self, analysis_results: Dict, video_name: str, 
                                output_dir: str) -> List[str]:
        """
        Create all available visualizations.
        
        Args:
            analysis_results: Results from trajectory analysis
            video_name: Name of the video
            output_dir: Output directory
            
        Returns:
            List of paths to saved plots
        """
        print("Creating all visualizations...")
        
        plot_paths = []
        
        # Create individual plots
        plot_paths.append(self.plot_trajectory(analysis_results, video_name, output_dir))
        plot_paths.append(self.plot_distance_analysis(analysis_results, video_name, output_dir))
        plot_paths.append(self.plot_cluster_analysis(analysis_results, video_name, output_dir))
        
        # Create event impact plot if events exist
        if analysis_results.get('event_stats'):
            plot_paths.append(self.plot_event_impact(analysis_results, video_name, output_dir))
        
        # Create summary plot
        plot_paths.append(self.create_summary_plot(analysis_results, video_name, output_dir))
        
        # Filter out empty paths
        plot_paths = [path for path in plot_paths if path]
        
        print(f"Created {len(plot_paths)} visualizations")
        return plot_paths


def main():
    """Test the visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test visualizer")
    parser.add_argument("--analysis-file", help="Path to analysis .npz file")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    parser.add_argument("--video-name", default="test_video", help="Video name")
    
    args = parser.parse_args()
    
    visualizer = EmbeddingVisualizer()
    
    if args.analysis_file:
        # Load analysis results
        data = np.load(args.analysis_file)
        analysis_results = {
            'embeddings_pca': data['embeddings_pca'],
            'distances': data['distances'],
            'similarities': data['similarities'],
            'cluster_labels': data['cluster_labels'],
            'explained_variance': np.array([0.4, 0.3, 0.2]),  # Dummy values
            'auto_events': [10, 20, 30]  # Dummy events
        }
    else:
        # Create dummy analysis results
        analysis_results = {
            'embeddings_pca': np.random.randn(64, 2),
            'distances': np.random.randn(63),
            'similarities': np.random.randn(63),
            'cluster_labels': np.random.randint(0, 5, 64),
            'explained_variance': np.array([0.4, 0.3]),
            'auto_events': [10, 20, 30]
        }
    
    # Create visualizations
    plot_paths = visualizer.create_all_visualizations(
        analysis_results, args.video_name, args.output_dir
    )
    
    print(f"Visualizations saved to: {plot_paths}")


if __name__ == "__main__":
    main()
