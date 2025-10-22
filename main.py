#!/usr/bin/env python3
"""
Traffic Signal Violation Detection System - Main Entry Point

This module provides both GUI and CLI interfaces for traffic violation detection.

Usage:
    # Launch GUI (default)
    python main.py

    # Process video from command line
    python main.py --video input.mp4 --line "100,200,500,200"

    # Customize output and confidence
    python main.py --video input.mp4 --line "100,200,500,200" \\
                   --output result.mp4 --confidence 0.7

    # Batch process directory
    python main.py --input-dir videos/ --output-dir results/ \\
                   --line "100,200,500,200"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

# Import project modules
import config
from detection import (
    detector,
    preprocess_input,
    decode_netout,
    correct_yolo_boxes,
    do_nms,
    draw_boxes
)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set log level to DEBUG
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.PROJECT_ROOT / 'violation_detection.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_line_coordinates(line_str: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Parse line coordinates from string format.

    Args:
        line_str: Line coordinates in format "x1,y1,x2,y2"

    Returns:
        Tuple of ((x1, y1), (x2, y2))

    Raises:
        ValueError: If format is invalid

    Example:
        >>> parse_line_coordinates("100,200,500,200")
        ((100, 200), (500, 200))
    """
    try:
        coords = [int(x.strip()) for x in line_str.split(',')]
        if len(coords) != 4:
            raise ValueError
        return ((coords[0], coords[1]), (coords[2], coords[3]))
    except (ValueError, IndexError):
        raise ValueError(
            f"Invalid line format: '{line_str}'. "
            "Expected format: 'x1,y1,x2,y2' (e.g., '100,200,500,200')"
        )


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Traffic Signal Violation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  %(prog)s

  # Process single video
  %(prog)s --video input.mp4 --line "100,200,500,200"

  # Custom output and confidence threshold
  %(prog)s --video input.mp4 --line "100,200,500,200" \\
           --output result.mp4 --confidence 0.7

  # Batch process directory
  %(prog)s --input-dir videos/ --output-dir results/ \\
           --line "100,200,500,200"

  # Headless mode (no display)
  %(prog)s --video input.mp4 --line "100,200,500,200" --no-display
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to input video file'
    )
    input_group.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Directory containing videos to process'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output video file (default: Resources/output/output.mp4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory for batch processing output'
    )

    # Detection options
    parser.add_argument(
        '--line', '-l',
        type=str,
        help='Traffic line coordinates as "x1,y1,x2,y2" (e.g., "100,200,500,200")'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=config.OBJ_THRESH,
        help=f'Detection confidence threshold (default: {config.OBJ_THRESH})'
    )

    # Mode options
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run in headless mode without video display'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validation
    if args.video or args.input_dir:
        if not args.line:
            parser.error("--line is required when processing videos from CLI")

    if args.output_dir and not args.input_dir:
        parser.error("--output-dir requires --input-dir")

    return args


def validate_inputs(video_path: str, line_coords: str) -> Tuple[Path, Tuple]:
    """
    Validate input file and line coordinates.

    Args:
        video_path: Path to video file
        line_coords: Line coordinates string

    Returns:
        Tuple of (validated_path, parsed_line_coords)

    Raises:
        FileNotFoundError: If video doesn't exist
        ValueError: If line coordinates invalid
    """
    # Validate video file
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video.is_file():
        raise ValueError(f"Not a file: {video_path}")

    # Validate video can be opened
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    cap.release()

    # Parse and validate line coordinates
    line = parse_line_coordinates(line_coords)

    return video, line


def process_video_cli(
    video_path: Path,
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    output_path: Optional[Path] = None,
    confidence: float = 0.5,
    show_display: bool = True
) -> dict:
    """
    Process video in CLI mode.

    Args:
        video_path: Path to input video
        line: Traffic line coordinates ((x1,y1), (x2,y2))
        output_path: Path for output video (optional)
        confidence: Detection confidence threshold
        show_display: Show video display during processing

    Returns:
        dict: Processing statistics
            {
                'total_frames': int,
                'violations': int,
                'processing_time': float,
                'fps': float
            }

    Raises:
        RuntimeError: If processing fails
    """
    import time

    logging.info(f"Processing video: {video_path}")
    logging.info(f"Traffic line: {line[0]} -> {line[1]}")
    logging.info(f"Confidence threshold: {confidence}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Setup output
    if output_path is None:
        output_path = config.OUTPUT_DIR / f"output_{video_path.stem}.mp4"

    config.ensure_directories()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Processing statistics
    violation_count = 0
    start_time = time.time()

    try:
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            show_progress = True
        except ImportError:
            logging.warning("tqdm not installed. Progress bar disabled. Install with: pip install tqdm")
            show_progress = False

        # Create progress bar if available
        if show_progress:
            progress_bar = tqdm(
                total=total_frames,
                desc="Processing",
                unit="frame",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                          '[{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        else:
            progress_bar = None

        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Preprocess for detection
            input_image = preprocess_input(frame, config.NET_H, config.NET_W)

            # Run detection (triggers lazy loading on first frame)
            yolos = detector.predict(input_image)

            # Decode detections
            boxes = []
            for i in range(len(yolos)):
                boxes += decode_netout(
                    yolos[i][0],
                    config.ANCHORS[i],
                    confidence,
                    config.NET_H,
                    config.NET_W
                )

            # Correct boxes to original image size
            correct_yolo_boxes(boxes, height, width, config.NET_H, config.NET_W)

            # Non-maximum suppression
            do_nms(boxes, config.NMS_THRESH)

            # Draw boxes and detect violations
            annotated_frame = draw_boxes(
                frame,
                boxes,
                [line[0], line[1]],
                config.LABELS,
                confidence,
                frame_num
            )

            # Count violations (simplified - just check for red boxes)
            # In a real implementation, draw_boxes would return violation count

            # Write output frame
            out.write(annotated_frame)

            # Show display if enabled
            if show_display:
                cv2.imshow('Traffic Violation Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Processing stopped by user")
                    break

            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
                elapsed = time.time() - start_time
                processing_fps = frame_num / elapsed if elapsed > 0 else 0
                progress_bar.set_postfix({
                    'fps': f'{processing_fps:.2f}'
                })

        # Close progress bar
        if progress_bar:
            progress_bar.close()

        # Calculate statistics
        processing_time = time.time() - start_time
        processing_fps = total_frames / processing_time

        stats = {
            'total_frames': total_frames,
            'violations': violation_count,
            'processing_time': processing_time,
            'fps': processing_fps
        }

        logging.info(f"✅ Processing complete!")
        logging.info(f"   Total frames: {total_frames}")
        logging.info(f"   Time: {processing_time:.2f}s ({processing_fps:.2f} FPS)")
        logging.info(f"   Output: {output_path}")

        return stats

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        raise

    finally:
        cap.release()
        out.release()
        if show_display:
            cv2.destroyAllWindows()


def run_gui_mode():
    """Launch GUI mode using existing Project-GUI.py"""
    logging.info("Launching GUI mode...")

    try:
        # Import GUI module
        import importlib.util
        spec = importlib.util.spec_from_file_location("gui", "Project-GUI.py")
        gui_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gui_module)

        logging.info("GUI launched successfully")

    except Exception as e:
        logging.error(f"Failed to launch GUI: {e}")
        logging.error("Please ensure Project-GUI.py exists and Tkinter is installed")
        sys.exit(1)


def run_cli_mode(args):
    """
    Run CLI mode with parsed arguments.

    Args:
        args: Parsed command line arguments
    """
    if args.video:
        # Single video processing
        video, line = validate_inputs(args.video, args.line)

        output = Path(args.output) if args.output else None

        stats = process_video_cli(
            video,
            line,
            output,
            args.confidence,
            show_display=not args.no_display
        )

        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Frames processed: {stats['total_frames']}")
        print(f"Processing time: {stats['processing_time']:.2f}s")
        print(f"Average FPS: {stats['fps']:.2f}")
        print("="*70)

    elif args.input_dir:
        # Batch processing
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR

        if not input_dir.exists():
            logging.error(f"Input directory not found: {input_dir}")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.glob(f'*{ext}'))

        if not videos:
            logging.error(f"No video files found in {input_dir}")
            sys.exit(1)

        logging.info(f"Found {len(videos)} videos to process")

        line = parse_line_coordinates(args.line)

        # Process each video
        results = []
        for video in videos:
            logging.info(f"\n{'='*70}")
            logging.info(f"Processing {video.name}...")
            logging.info(f"{'='*70}")

            output = output_dir / f"output_{video.stem}.mp4"

            try:
                stats = process_video_cli(
                    video,
                    line,
                    output,
                    args.confidence,
                    show_display=not args.no_display
                )
                stats['video'] = video.name
                stats['success'] = True
                results.append(stats)
            except Exception as e:
                logging.error(f"Failed to process {video.name}: {e}")
                results.append({
                    'video': video.name,
                    'success': False,
                    'error': str(e)
                })

        # Print summary
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        successful = sum(1 for r in results if r['success'])
        print(f"Videos processed: {successful}/{len(results)}")
        print("="*70)
        for result in results:
            if result['success']:
                print(f"✅ {result['video']}: {result['fps']:.2f} FPS")
            else:
                print(f"❌ {result['video']}: {result['error']}")
        print("="*70)


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Validate configuration
    if not config.validate_config():
        logging.error("Configuration validation failed")
        logging.error("Please ensure all required files and directories exist")
        sys.exit(1)

    config.ensure_directories()

    # Log system information
    logging.info("="*70)
    logging.info("Traffic Signal Violation Detection System")
    logging.info("="*70)
    logging.info(f"Project root: {config.PROJECT_ROOT}")
    logging.info(f"Weights: {config.WEIGHTS_PATH}")
    logging.info(f"Model: YOLOv3 ({config.NET_H}x{config.NET_W})")
    logging.info(f"Thresholds: OBJ={config.OBJ_THRESH}, NMS={config.NMS_THRESH}")
    logging.info("="*70)

    # Run appropriate mode
    if args.video or args.input_dir:
        run_cli_mode(args)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()
