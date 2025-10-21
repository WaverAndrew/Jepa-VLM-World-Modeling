#!/usr/bin/env python3
"""
Video Processing Module

Handles video loading, frame sampling, and preprocessing for V-JEPA2 analysis.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import decord
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image


class VideoProcessor:
    """Handles video loading and frame preprocessing."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize video processor.
        
        Args:
            target_size: Target frame size (width, height)
        """
        self.target_size = target_size
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_video(self, video_path: str, num_frames: int = 64) -> np.ndarray:
        """
        Load and uniformly sample frames from a video.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            
        Returns:
            Array of frames with shape (num_frames, height, width, channels)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Loading video: {video_path}")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames < num_frames:
                print(f"Warning: Video has only {total_frames} frames, using all frames")
                indices = np.arange(total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = vr.get_batch(indices).asnumpy()
            print(f"Sampled {len(frames)} frames from video")
            
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Error loading video: {e}")
    
    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess frames for model input.
        
        Args:
            frames: Array of frames (num_frames, height, width, channels)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        print("Preprocessing frames...")
        
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
        
        # Stack frames and add batch dimension
        frames_tensor = torch.stack(processed_frames).unsqueeze(0)  # [1, num_frames, C, H, W]
        
        print(f"Preprocessed frames shape: {frames_tensor.shape}")
        return frames_tensor
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about the video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            
            info = {
                'path': video_path,
                'name': Path(video_path).stem,
                'total_frames': len(vr),
                'fps': vr.get_avg_fps(),
                'duration': len(vr) / vr.get_avg_fps() if vr.get_avg_fps() > 0 else 0,
                'width': vr[0].shape[1],
                'height': vr[0].shape[0]
            }
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Error getting video info: {e}")
    
    def save_sample_frames(self, frames: np.ndarray, output_dir: str, 
                          video_name: str, num_samples: int = 5) -> List[str]:
        """
        Save sample frames for inspection.
        
        Args:
            frames: Array of frames
            output_dir: Output directory
            video_name: Name of the video
            num_samples: Number of sample frames to save
            
        Returns:
            List of saved frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample frames evenly
        if len(frames) <= num_samples:
            indices = range(len(frames))
        else:
            indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
        
        saved_paths = []
        for i, frame_idx in enumerate(indices):
            frame = frames[frame_idx]
            
            # Convert to PIL Image and save
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(frame)
            frame_path = f"{output_dir}/{video_name}_frame_{i:02d}.png"
            image.save(frame_path)
            saved_paths.append(frame_path)
        
        print(f"Saved {len(saved_paths)} sample frames to {output_dir}/")
        return saved_paths


def main():
    """Test the video processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video processor")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    parser.add_argument("--num-frames", type=int, default=64, help="Number of frames to sample")
    
    args = parser.parse_args()
    
    processor = VideoProcessor()
    
    # Get video info
    info = processor.get_video_info(args.video)
    print("Video Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Load and preprocess video
    frames = processor.load_video(args.video, args.num_frames)
    processed_tensor = processor.preprocess_frames(frames)
    
    # Save sample frames
    saved_paths = processor.save_sample_frames(frames, args.output_dir, info['name'])
    
    print(f"Processing complete! Sample frames saved to: {saved_paths}")


if __name__ == "__main__":
    main()
