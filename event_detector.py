#!/usr/bin/env python3
"""
Event Detection Module

Handles both manual event annotation and automatic event detection.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class EventDetector:
    """Handles event detection and annotation."""
    
    def __init__(self, motion_threshold: float = 30.0):
        """
        Initialize event detector.
        
        Args:
            motion_threshold: Threshold for automatic motion detection
        """
        self.motion_threshold = motion_threshold
    
    def load_manual_events(self, events_path: str) -> Dict[str, Tuple[int, int]]:
        """
        Load manually annotated events from JSON file.
        
        Args:
            events_path: Path to JSON file with event annotations
            
        Returns:
            Dictionary mapping event names to (start_frame, end_frame) tuples
        """
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"Events file not found: {events_path}")
        
        try:
            with open(events_path, 'r') as f:
                events_data = json.load(f)
            
            events = {}
            for event_name, frame_range in events_data.items():
                if isinstance(frame_range, list) and len(frame_range) == 2:
                    events[event_name] = (frame_range[0], frame_range[1])
                else:
                    print(f"Warning: Invalid format for event '{event_name}': {frame_range}")
            
            print(f"Loaded {len(events)} manual events: {list(events.keys())}")
            return events
            
        except Exception as e:
            raise RuntimeError(f"Error loading events file: {e}")
    
    def detect_motion_events(self, frames: np.ndarray, 
                           threshold: Optional[float] = None) -> List[int]:
        """
        Detect events using motion-based detection.
        
        Args:
            frames: Array of video frames
            threshold: Motion threshold (uses instance default if None)
            
        Returns:
            List of frame indices where events are detected
        """
        threshold = threshold or self.motion_threshold
        print(f"Detecting motion events with threshold: {threshold}")
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        # Compute frame differences
        motion_scores = []
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
            motion_score = np.sum(diff)
            motion_scores.append(motion_score)
        
        # Find frames exceeding threshold
        events = [i for i, motion in enumerate(motion_scores) if motion > threshold]
        
        print(f"Detected {len(events)} motion events at frames: {events}")
        
        # Return motion scores for analysis
        return events, motion_scores
    
    def detect_scene_changes(self, frames: np.ndarray, 
                            threshold: float = 0.3) -> List[int]:
        """
        Detect scene changes using histogram comparison.
        
        Args:
            frames: Array of video frames
            threshold: Threshold for scene change detection
            
        Returns:
            List of frame indices where scene changes are detected
        """
        print(f"Detecting scene changes with threshold: {threshold}")
        
        scene_changes = []
        prev_hist = None
        
        for i, frame in enumerate(frames):
            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compare histograms using correlation
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                if correlation < threshold:
                    scene_changes.append(i)
            
            prev_hist = hist
        
        print(f"Detected {len(scene_changes)} scene changes at frames: {scene_changes}")
        return scene_changes
    
    def detect_object_motion(self, frames: np.ndarray, 
                           min_area: int = 1000) -> List[int]:
        """
        Detect object motion using optical flow.
        
        Args:
            frames: Array of video frames
            min_area: Minimum area for motion detection
            
        Returns:
            List of frame indices with significant object motion
        """
        print(f"Detecting object motion with min area: {min_area}")
        
        motion_frames = []
        prev_gray = None
        
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, 
                    np.array([[100, 100]], dtype=np.float32), 
                    None
                )[0]
                
                # Check for significant motion
                if flow is not None and len(flow) > 0:
                    motion_magnitude = np.linalg.norm(flow)
                    if motion_magnitude > min_area:
                        motion_frames.append(i)
            
            prev_gray = gray
        
        print(f"Detected {len(motion_frames)} frames with object motion: {motion_frames}")
        return motion_frames
    
    def combine_event_detections(self, motion_events: List[int], 
                               scene_changes: List[int],
                               object_motion: List[int]) -> Dict[str, List[int]]:
        """
        Combine different types of event detections.
        
        Args:
            motion_events: Motion-based events
            scene_changes: Scene change events
            object_motion: Object motion events
            
        Returns:
            Dictionary with combined event information
        """
        all_events = set(motion_events + scene_changes + object_motion)
        
        combined = {
            'all_events': sorted(list(all_events)),
            'motion_events': motion_events,
            'scene_changes': scene_changes,
            'object_motion': object_motion,
            'motion_and_scene': list(set(motion_events) & set(scene_changes)),
            'motion_and_object': list(set(motion_events) & set(object_motion)),
            'scene_and_object': list(set(scene_changes) & set(object_motion))
        }
        
        print(f"Combined detection results:")
        for event_type, events in combined.items():
            print(f"  {event_type}: {len(events)} events")
        
        return combined
    
    def save_event_annotations(self, events: Dict[str, Union[List[int], Tuple[int, int]]], 
                             output_path: str):
        """
        Save event annotations to JSON file.
        
        Args:
            events: Event annotations dictionary
            output_path: Path to save the JSON file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to JSON-serializable format
        json_events = {}
        for event_name, event_data in events.items():
            if isinstance(event_data, tuple) and len(event_data) == 2:
                json_events[event_name] = list(event_data)
            elif isinstance(event_data, list):
                json_events[event_name] = event_data
            else:
                json_events[event_name] = str(event_data)
        
        with open(output_path, 'w') as f:
            json.dump(json_events, f, indent=2)
        
        print(f"Event annotations saved to: {output_path}")
    
    def create_event_timeline(self, events: Dict[str, Union[List[int], Tuple[int, int]]], 
                            total_frames: int) -> np.ndarray:
        """
        Create a binary timeline indicating event presence.
        
        Args:
            events: Event annotations
            total_frames: Total number of frames
            
        Returns:
            Binary array indicating event presence per frame
        """
        timeline = np.zeros(total_frames, dtype=bool)
        
        for event_name, event_data in events.items():
            if isinstance(event_data, tuple) and len(event_data) == 2:
                start, end = event_data
                timeline[start:end+1] = True
            elif isinstance(event_data, list):
                for frame_idx in event_data:
                    if 0 <= frame_idx < total_frames:
                        timeline[frame_idx] = True
        
        return timeline


def main():
    """Test the event detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test event detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--events", help="Path to manual events JSON file")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    parser.add_argument("--motion-threshold", type=float, default=30.0, 
                       help="Motion detection threshold")
    
    args = parser.parse_args()
    
    # Import video processor for testing
    from video_processor import VideoProcessor
    
    detector = EventDetector(motion_threshold=args.motion_threshold)
    processor = VideoProcessor()
    
    # Load video
    frames = processor.load_video(args.video, num_frames=64)
    
    # Load manual events if provided
    manual_events = {}
    if args.events:
        manual_events = detector.load_manual_events(args.events)
    
    # Detect events automatically
    motion_events, motion_scores = detector.detect_motion_events(frames)
    scene_changes = detector.detect_scene_changes(frames)
    object_motion = detector.detect_object_motion(frames)
    
    # Combine detections
    combined = detector.combine_event_detections(motion_events, scene_changes, object_motion)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    detector.save_event_annotations(combined, f"{args.output_dir}/detected_events.json")
    
    if manual_events:
        detector.save_event_annotations(manual_events, f"{args.output_dir}/manual_events.json")
    
    print("Event detection test complete!")


if __name__ == "__main__":
    main()
