#!/usr/bin/env python3
"""
V-JEPA2 Model Wrapper Module

Handles V-JEPA2 model loading, CUDA optimization, and embedding extraction.
"""

import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoVideoProcessor
import numpy as np


class VJEPA2Model:
    """Wrapper for V-JEPA2 model with CUDA optimization."""
    
    def __init__(self, model_name: str = "facebook/vjepa2-vith-fpc64-256", 
                 device: Optional[str] = None):
        """
        Initialize V-JEPA2 model wrapper.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or self._get_device()
        self.model = None
        self.processor = None
        self.scaler = None
        
        print(f"Using device: {self.device}")
        self._load_model()
    
    def _get_device(self) -> str:
        """Automatically detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the V-JEPA2 model and processor."""
        print(f"Loading V-JEPA2 model: {self.model_name}")
        
        try:
            # Load model with appropriate dtype
            if self.device == "cuda":
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            
            # Load processor
            self.processor = AutoVideoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Enable CUDA optimizations
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.scaler = torch.cuda.amp.GradScaler()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_video(self, frames_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Preprocess video frames for V-JEPA2 model.
        
        Args:
            frames_tensor: Preprocessed frames tensor [1, num_frames, C, H, W]
            
        Returns:
            Dictionary with model inputs
        """
        print("Preprocessing video for V-JEPA2...")
        
        # Convert tensor to numpy for processor (expects uint8 RGB frames [T, H, W, C])
        frames_np = frames_tensor.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
        # If values look normalized to 0-1, rescale to 0-255
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255.0).clip(0, 255)
        frames_np = frames_np.astype(np.uint8)

        # Use processor (it may return different key names across versions)
        inputs = self.processor(frames_np, return_tensors="pt")
        
        # Move to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Best-effort report of primary input tensor shape
        key = self._get_primary_input_key(inputs)
        if key is not None:
            print(f"Preprocessed inputs shape ({key}): {inputs[key].shape}")
        else:
            print(f"Preprocessed inputs keys: {list(inputs.keys())}")
        return inputs

    def _get_primary_input_key(self, inputs: Dict[str, torch.Tensor]) -> Optional[str]:
        """Return the canonical tensor key used by the processor/model."""
        candidate_keys = [
            'pixel_values',              # common for video/image processors
            'video_values',              # some video processors
            'pixel_values_videos',       # alternative naming
            'videos',                    # rare
            'inputs',                    # generic
        ]
        for k in candidate_keys:
            if k in inputs and isinstance(inputs[k], torch.Tensor):
                return k
        # fallback: first tensor value
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                return k
        return None
    
    def extract_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract embeddings from V-JEPA2 encoder.
        
        Args:
            inputs: Preprocessed model inputs
            
        Returns:
            Embeddings tensor with shape [batch_size, num_patches, embedding_dim]
        """
        print("Extracting embeddings...")
        
        with torch.no_grad():
            if self.device == "cuda" and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            # Extract last hidden state
            embeddings = outputs.last_hidden_state
            
            print(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_dtype': next(self.model.parameters()).dtype,
            'mixed_precision': self.scaler is not None
        }
        
        return info
    
    def benchmark_inference(self, inputs: Dict[str, torch.Tensor], 
                          num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            inputs: Model inputs
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"Benchmarking inference for {num_runs} runs...")
        
        # Warmup
        with torch.no_grad():
            _ = self.model(**inputs)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                if self.device == "cuda" and self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        _ = self.model(**inputs)
                else:
                    _ = self.model(**inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        stats = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }
        
        print(f"Benchmark results: {stats['mean_time']:.3f}s ± {stats['std_time']:.3f}s")
        print(f"Throughput: {stats['fps']:.1f} FPS")
        
        return stats


def main():
    """Test the V-JEPA2 model wrapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test V-JEPA2 model wrapper")
    parser.add_argument("--model", default="facebook/vjepa2-vith-fpc64-256", 
                       help="Model name")
    parser.add_argument("--device", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize model
    model_wrapper = VJEPA2Model(model_name=args.model, device=args.device)
    
    # Get model info
    info = model_wrapper.get_model_info()
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dummy input for testing
    dummy_frames = torch.randn(1, 64, 3, 256, 256)
    inputs = model_wrapper.preprocess_video(dummy_frames)
    
    # Extract embeddings
    embeddings = model_wrapper.extract_embeddings(inputs)
    
    # Benchmark
    benchmark_stats = model_wrapper.benchmark_inference(inputs, num_runs=5)
    
    print("Test complete!")


if __name__ == "__main__":
    main()
