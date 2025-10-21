#!/usr/bin/env python3
"""
Main Pipeline Script

Orchestrates the complete V-JEPA2 video embedding analysis pipeline.
This is the main entry point for the analysis system.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import our modules
from embedding_analyzer import EmbeddingAnalyzer
from event_detector import EventDetector
from vjepa_model import VJEPA2Model
from video_processor import VideoProcessor
from visualizer import EmbeddingVisualizer


class VJEPA2Pipeline:
    """Main pipeline for V-JEPA2 video embedding analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.video_processor = None
        self.model_wrapper = None
        self.event_detector = None
        self.embedding_analyzer = None
        self.visualizer = None
        
        self.results = {}
    
    def setup_components(self, model_name: str = "facebook/vjepa2-vith-fpc64-256",
                       device: Optional[str] = None,
                       motion_threshold: float = 30.0,
                       n_clusters: int = 5):
        """
        Setup all pipeline components.
        
        Args:
            model_name: V-JEPA2 model name
            device: Device to use
            motion_threshold: Motion detection threshold
            n_clusters: Number of clusters for analysis
        """
        print("Setting up pipeline components...")
        
        # Initialize components
        self.video_processor = VideoProcessor()
        self.model_wrapper = VJEPA2Model(model_name=model_name, device=device)
        self.event_detector = EventDetector(motion_threshold=motion_threshold)
        self.embedding_analyzer = EmbeddingAnalyzer(n_clusters=n_clusters)
        self.visualizer = EmbeddingVisualizer()
        
        print("Pipeline components ready!")
    
    def process_video(self, video_path: str, num_frames: int = 64) -> Dict:
        """
        Process a video through the complete pipeline.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            
        Returns:
            Dictionary with all analysis results
        """
        print(f"Processing video: {video_path}")
        
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)
        video_name = video_info['name']
        
        print(f"Video: {video_name}")
        print(f"Duration: {video_info['duration']:.2f}s")
        print(f"Frames: {video_info['total_frames']}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        
        # Step 1: Load and preprocess video
        print("\n=== Step 1: Video Processing ===")
        frames = self.video_processor.load_video(video_path, num_frames)
        frames_tensor = self.video_processor.preprocess_frames(frames)
        
        # Step 2: Extract embeddings
        print("\n=== Step 2: Embedding Extraction ===")
        model_inputs = self.model_wrapper.preprocess_video(frames_tensor)
        embeddings = self.model_wrapper.extract_embeddings(model_inputs)
        
        # Step 3: Event detection
        print("\n=== Step 3: Event Detection ===")
        manual_events = {}
        auto_events = None
        
        # Load manual events if provided
        if 'events_file' in self.config and self.config['events_file']:
            manual_events = self.event_detector.load_manual_events(self.config['events_file'])
        
        # Detect events automatically if enabled
        if self.config.get('auto_detect_events', False):
            auto_events, motion_scores = self.event_detector.detect_motion_events(frames)
        
        # Step 4: Embedding analysis
        print("\n=== Step 4: Embedding Analysis ===")
        analysis_results = self.embedding_analyzer.analyze_trajectory(
            embeddings, manual_events, auto_events
        )
        
        # Add additional analysis
        embedding_stats = self.embedding_analyzer.compute_embedding_statistics(embeddings)
        temporal_results = self.embedding_analyzer.analyze_temporal_dynamics(embeddings)
        similarity_matrix = self.embedding_analyzer.compute_embedding_similarity_matrix(embeddings)
        
        # Step 5: Visualization
        print("\n=== Step 5: Visualization ===")
        output_dir = self.config.get('output_dir', 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = self.visualizer.create_all_visualizations(
            analysis_results, video_name, output_dir
        )
        
        # Create similarity heatmap
        heatmap_path = self.visualizer.plot_heatmap(
            similarity_matrix, video_name, output_dir
        )
        if heatmap_path:
            plot_paths.append(heatmap_path)
        
        # Step 6: Export results
        print("\n=== Step 6: Export Results ===")
        self._export_results(
            video_info, analysis_results, embedding_stats, temporal_results,
            manual_events, auto_events, plot_paths, output_dir
        )
        
        # Store results
        self.results = {
            'video_info': video_info,
            'analysis_results': analysis_results,
            'embedding_stats': embedding_stats,
            'temporal_results': temporal_results,
            'manual_events': manual_events,
            'auto_events': auto_events,
            'plot_paths': plot_paths,
            'model_info': self.model_wrapper.get_model_info()
        }
        
        print(f"\nPipeline complete! Results saved to {output_dir}/")
        return self.results
    
    def _export_results(self, video_info: Dict, analysis_results: Dict,
                      embedding_stats: Dict, temporal_results: Dict,
                      manual_events: Dict, auto_events: Optional[List[int]],
                      plot_paths: List[str], output_dir: str):
        """Export all results to files."""
        
        video_name = video_info['name']
        
        # Export analysis results
        self.embedding_analyzer.export_analysis_results(
            analysis_results, f"{output_dir}/{video_name}_analysis"
        )
        
        # Export event annotations
        if manual_events:
            self.event_detector.save_event_annotations(
                manual_events, f"{output_dir}/{video_name}_manual_events.json"
            )
        
        if auto_events:
            auto_events_dict = {'auto_detected_events': auto_events}
            self.event_detector.save_event_annotations(
                auto_events_dict, f"{output_dir}/{video_name}_auto_events.json"
            )
        
        # Export comprehensive results JSON
        comprehensive_results = {
            'video_info': video_info,
            'model_info': self.model_wrapper.get_model_info(),
            'embedding_stats': embedding_stats,
            'temporal_results': temporal_results,
            'manual_events': manual_events,
            'auto_events': auto_events,
            'plot_paths': plot_paths,
            'analysis_summary': {
                'num_frames_analyzed': len(analysis_results['embeddings_pca']),
                'num_clusters': len(np.unique(analysis_results['cluster_labels'])),
                'num_manual_events': len(manual_events),
                'num_auto_events': len(auto_events) if auto_events else 0,
                'explained_variance_pc1': float(analysis_results['explained_variance'][0]),
                'explained_variance_pc2': float(analysis_results['explained_variance'][1])
            }
        }
        
        with open(f"{output_dir}/{video_name}_comprehensive_results.json", 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Export frame-level CSV
        frame_data = []
        for i in range(len(analysis_results['embeddings_pca'])):
            frame_data.append({
                'frame_index': i,
                'pc1': analysis_results['embeddings_pca'][i, 0],
                'pc2': analysis_results['embeddings_pca'][i, 1],
                'cluster_id': analysis_results['cluster_labels'][i],
                'distance_to_next': analysis_results['distances'][i] if i < len(analysis_results['distances']) else None,
                'similarity_to_next': analysis_results['similarities'][i] if i < len(analysis_results['similarities']) else None,
                'is_auto_event': i in (auto_events or []),
                'velocity_magnitude': temporal_results['velocity_magnitude'][i] if i < len(temporal_results['velocity_magnitude']) else None,
                'acceleration_magnitude': temporal_results['acceleration_magnitude'][i] if i < len(temporal_results['acceleration_magnitude']) else None
            })
        
        df = pd.DataFrame(frame_data)
        df.to_csv(f"{output_dir}/{video_name}_frame_data.csv", index=False)
        
        print(f"Results exported to {output_dir}/")
    
    def benchmark_pipeline(self, video_path: str, num_runs: int = 3) -> Dict:
        """
        Benchmark the pipeline performance.
        
        Args:
            video_path: Path to test video
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking pipeline with {num_runs} runs...")
        
        import time
        
        times = []
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            start_time = time.time()
            self.process_video(video_path)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        benchmark_stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'times': times
        }
        
        print(f"Benchmark results:")
        print(f"  Mean time: {benchmark_stats['mean_time']:.2f}s")
        print(f"  Std time: {benchmark_stats['std_time']:.2f}s")
        print(f"  Min time: {benchmark_stats['min_time']:.2f}s")
        print(f"  Max time: {benchmark_stats['max_time']:.2f}s")
        
        return benchmark_stats


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="V-JEPA2 Video Embedding Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python main.py --video path/to/video.mp4
  
  # Full analysis with events
  python main.py --video path/to/video.mp4 --events events.json --auto-detect-events
  
  # Custom configuration
  python main.py --video path/to/video.mp4 --num-frames 128 --output-dir results/
        """
    )
    
    # Required arguments
    parser.add_argument("--video", required=True, help="Path to input video file")
    
    # Optional arguments
    parser.add_argument("--events", help="Path to JSON file with manual event annotations")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--auto-detect-events", action="store_true", 
                       help="Enable automatic event detection")
    parser.add_argument("--num-frames", type=int, default=64, 
                       help="Number of frames to sample from video")
    parser.add_argument("--motion-threshold", type=float, default=30.0,
                       help="Threshold for automatic event detection")
    parser.add_argument("--model", default="facebook/vjepa2-vith-fpc64-256",
                       help="V-JEPA2 model name")
    parser.add_argument("--device", help="Device to use (cuda/cpu/mps)")
    parser.add_argument("--n-clusters", type=int, default=5,
                       help="Number of clusters for analysis")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark mode")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    config.update({
        'video_path': args.video,
        'events_file': args.events,
        'output_dir': args.output_dir,
        'auto_detect_events': args.auto_detect_events,
        'num_frames': args.num_frames,
        'motion_threshold': args.motion_threshold,
        'model_name': args.model,
        'device': args.device,
        'n_clusters': args.n_clusters
    })
    
    # Initialize pipeline
    pipeline = VJEPA2Pipeline(config)
    
    # Setup components
    pipeline.setup_components(
        model_name=config['model_name'],
        device=config['device'],
        motion_threshold=config['motion_threshold'],
        n_clusters=config['n_clusters']
    )
    
    # Run pipeline
    if args.benchmark:
        benchmark_results = pipeline.benchmark_pipeline(config['video_path'])
        
        # Save benchmark results
        with open(f"{config['output_dir']}/benchmark_results.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    else:
        results = pipeline.process_video(
            config['video_path'], 
            config['num_frames']
        )
        
        print("\n=== Analysis Complete ===")
        print(f"Video: {results['video_info']['name']}")
        print(f"Frames analyzed: {results['analysis_results']['embeddings_pca'].shape[0]}")
        print(f"Manual events: {len(results['manual_events'])}")
        print(f"Auto-detected events: {len(results['auto_events']) if results['auto_events'] else 0}")
        print(f"Visualizations created: {len(results['plot_paths'])}")
        print(f"Results saved to: {config['output_dir']}/")


if __name__ == "__main__":
    main()
