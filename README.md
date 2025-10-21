# V-JEPA2 Video Embedding Analysis

A modular Python framework for analyzing video embeddings using the V-JEPA2 ViT-Huge model (600M parameters). The system processes videos through the V-JEPA2 encoder and analyzes how embeddings respond to specific events through comprehensive geometric analysis.

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd vjepa_encoder
pip install -r requirements.txt
```

### 2. Place Your Video

Put your video file in the project directory or provide the full path:

```bash
# Example: place video.mp4 in the project root
cp /path/to/your/video.mp4 ./video.mp4
```

### 3. Run Analysis

```bash
# Basic analysis
python main.py --video video.mp4

# Full analysis with events
python main.py --video video.mp4 --events events_example.json --auto-detect-events
```

## 📁 Project Structure

```
vjepa_encoder/
├── main.py                 # 🎯 MAIN ENTRY POINT - Start here!
├── video_processor.py      # Video loading and preprocessing
├── vjepa_model.py         # V-JEPA2 model wrapper with CUDA optimization
├── event_detector.py      # Event detection (manual + automatic)
├── embedding_analyzer.py   # Geometric analysis of embeddings
├── visualizer.py          # Publication-ready visualizations
├── requirements.txt       # Dependencies
├── config_example.json    # Example configuration
├── events_example.json    # Example event annotations
└── README.md             # This file
```

## 🎯 Where to Start

**Start with `main.py`** - This is the main entry point that orchestrates the entire pipeline.

### Basic Usage

```bash
python main.py --video path/to/your/video.mp4
```

### Advanced Usage

```bash
python main.py \
  --video path/to/your/video.mp4 \
  --events events_example.json \
  --output-dir results/ \
  --auto-detect-events \
  --num-frames 64
```

## 🔧 Modular Components

### Individual Module Usage

Each module can be used independently for specific tasks:

#### Video Processing

```bash
python video_processor.py --video video.mp4 --output-dir test_output
```

#### Model Testing

```bash
python vjepa_model.py --model facebook/vjepa2-vith-fpc64-256
```

#### Event Detection

```bash
python event_detector.py --video video.mp4 --motion-threshold 30.0
```

#### Embedding Analysis

```bash
python embedding_analyzer.py --embeddings embeddings.npz
```

#### Visualization

```bash
python visualizer.py --analysis-file analysis.npz --video-name test_video
```

## ⚙️ Configuration

### Command Line Arguments

- `--video`: Path to input video file (required)
- `--events`: Path to JSON file with manual event annotations
- `--output-dir`: Output directory for results (default: "results")
- `--auto-detect-events`: Enable automatic event detection
- `--num-frames`: Number of frames to sample (default: 64)
- `--motion-threshold`: Motion detection threshold (default: 30.0)
- `--model`: V-JEPA2 model name (default: "facebook/vjepa2-vith-fpc64-256")
- `--device`: Device to use (cuda/cpu/mps - auto-detected)
- `--n-clusters`: Number of clusters for analysis (default: 5)
- `--benchmark`: Run benchmark mode
- `--config`: Path to configuration JSON file

### Configuration File

Use `config_example.json` as a template:

```json
{
  "model_name": "facebook/vjepa2-vith-fpc64-256",
  "device": "cuda",
  "num_frames": 64,
  "motion_threshold": 30.0,
  "n_clusters": 5,
  "auto_detect_events": true,
  "output_dir": "results"
}
```

## 📊 Event Annotation

### Manual Events

Create a JSON file with event annotations:

```json
{
  "person_enters": [10, 20],
  "object_falls": [35, 45],
  "scene_change": [60, 70]
}
```

Each event maps to `[start_frame, end_frame]` range.

### Automatic Detection

The system automatically detects:

- Motion events (frame differencing)
- Scene changes (histogram comparison)
- Object motion (optical flow)

## 📈 Output Files

### Visualizations

- `{video_name}_trajectory.png`: PCA trajectory plot with cluster coloring
- `{video_name}_distances.png`: Inter-frame distance and similarity plots
- `{video_name}_clusters.png`: Cluster assignment over time
- `{video_name}_event_impact.png`: Event impact analysis
- `{video_name}_similarity_heatmap.png`: Similarity matrix heatmap
- `{video_name}_summary.png`: Comprehensive summary plot

### Data Files

- `{video_name}_comprehensive_results.json`: Complete analysis results
- `{video_name}_frame_data.csv`: Frame-level embedding data
- `{video_name}_analysis.json`: Analysis summary
- `{video_name}_data.npz`: Raw analysis data

## 🔬 Analysis Methods

### Trajectory Analysis

- Euclidean distances between consecutive frames
- Cosine similarities for angular changes
- PCA visualization of embedding evolution

### Clustering

- K-means clustering to identify video segments
- Analysis of cluster transitions during events

### Event Impact Analysis

- Statistical comparison of event vs non-event frames
- Magnitude ratios and distance metrics
- Temporal dynamics analysis

### Advanced Features

- Anomaly detection in embeddings
- Similarity matrix computation
- Velocity and acceleration analysis

## 🐍 Python API

### Programmatic Usage

```python
from main import VJEPA2Pipeline

# Initialize pipeline
pipeline = VJEPA2Pipeline()

# Setup components
pipeline.setup_components(
    model_name="facebook/vjepa2-vith-fpc64-256",
    device="cuda",
    motion_threshold=30.0
)

# Process video
results = pipeline.process_video("video.mp4", num_frames=64)

# Access results
print(f"Analyzed {len(results['analysis_results']['embeddings_pca'])} frames")
print(f"Detected {len(results['auto_events'])} events")
```

### Individual Module Usage

```python
from video_processor import VideoProcessor
from vjepa_model import VJEPA2Model
from event_detector import EventDetector
from embedding_analyzer import EmbeddingAnalyzer
from visualizer import EmbeddingVisualizer

# Use individual components
processor = VideoProcessor()
model = VJEPA2Model()
detector = EventDetector()
analyzer = EmbeddingAnalyzer()
visualizer = EmbeddingVisualizer()
```

## 🚀 Performance

### CUDA Optimization

- Automatic GPU detection with fallback to CPU/MPS
- Mixed precision (FP16) for faster inference
- cuDNN benchmarking for optimized convolutions
- Memory-efficient batch processing

### Benchmarking

```bash
python main.py --video video.mp4 --benchmark
```

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ GPU memory

## 🔗 References

- [V-JEPA2 Official Repository](https://github.com/facebookresearch/vjepa2)
- [V-JEPA2 Demo Notebook](https://github.com/facebookresearch/vjepa2/blob/main/notebooks/vjepa2_demo.ipynb)
- [HuggingFace Model](https://huggingface.co/facebook/vjepa2-vith-fpc64-256)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
