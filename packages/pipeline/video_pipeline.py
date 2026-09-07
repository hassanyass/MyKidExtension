"""
MyKid Video Pipeline.

Orchestrates the end-to-end processing of videos through the image pipeline.
"""
import logging
import time
from pathlib import Path
import cv2

from packages.shared.config import MyKidConfig
from packages.shared.types import AnalysisResult
from packages.shared.logger import get_logger, log_event, LogEvent
from packages.shared.errors import VideoProcessingError

from packages.video_processing.video_processor import open_video, get_metadata, create_writer
from packages.pipeline.image_pipeline import ImagePipeline


logger = get_logger("mykid.pipeline.video")


class VideoPipeline:
    """
    End-to-end video processing pipeline.
    Uses the ImagePipeline to process frames and re-encodes the video.
    """
    
    def __init__(self, config: MyKidConfig, image_pipeline: ImagePipeline):
        """
        Initialize the Video Pipeline.
        
        Args:
            config: Global configuration object.
            image_pipeline: An initialized ImagePipeline instance.
        """
        self.config = config
        self.image_pipeline = image_pipeline

    def process(self, input_path: str, output_path: str) -> None:
        """
        Process a video from start to finish, writing the protected video to output_path.
        
        Args:
            input_path: Path to the source video.
            output_path: Path where the protected video will be saved.
        """
        start_time = time.time()
        log_event(
            logger, 
            logging.INFO, 
            LogEvent.VIDEO_ANALYSIS_STARTED, 
            f"Started processing video: {input_path}"
        )
        
        cap = None
        writer = None
        
        try:
            # 1. Open Video and Read Metadata
            cap = open_video(input_path)
            meta = get_metadata(cap)
            log_event(logger, logging.DEBUG, LogEvent.VIDEO_METADATA_READ, "Video metadata", metadata=meta)
            
            # Enforce max resolution
            if meta["width"] > self.config.video.max_video_resolution or meta["height"] > self.config.video.max_video_resolution:
                raise VideoProcessingError(
                    f"Video exceeds maximum allowed resolution ({self.config.video.max_video_resolution})"
                )
                
            # 2. Setup Writer (same dimensions/fps as input)
            # Note: We output at the original resolution, but the image pipeline will resize internally 
            # for inference and return the *resized* protected image.
            # We need to decide if we upscale the protected image back to original resolution, or
            # just output the video at the smaller pipeline resolution.
            # For this MVP, let's output at the pipeline's resolution to save space, but
            # if we wanted original resolution, we'd upscale `protected_frame` before writing.
            
            # Setup writer will happen dynamically on the first processed frame
            

            
            # 3. Process frames
            frame_count = 0
            
            # Calculate inference interval (e.g. 30 fps video / 10 fps inference = run AI every 3 frames)
            inference_fps = self.config.video.inference_fps
            video_fps = meta["fps"]
            inference_interval = max(1, int(video_fps / inference_fps))
            
            from packages.video_processing.temporal_tracker import TemporalTracker
            tracker = TemporalTracker(max_persistence_frames=self.config.video.temporal_persistence_frames)
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                    
                # The video processor (loader equivalent for video) needs RGB for consistency 
                # before going to image_pipeline or tracker
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Check if it's an inference frame
                if frame_count % inference_interval == 0:
                    # Run full AI pipeline
                    protected_rgb, analysis = self.image_pipeline.process(frame_rgb)
                    # Initialize trackers with the raw frame (pre-blur) and the new analysis
                    tracker.initialize(frame_rgb, analysis)
                else:
                    # Non-inference frame: Update tracking
                    analysis = tracker.update(frame_rgb)
                    # Apply protection using the updated tracked coordinates
                    # The ProtectionEngine uses the same logic regardless of where the analysis came from
                    # Need to preprocess frame first if we're calling protection directly, 
                    # but wait! ImagePipeline handles preprocessing (resize). 
                    # To keep it simple and consistent for MVP, we'll just run the protection engine manually here,
                    # or better: we pass the image to preprocessor, then protect.
                    # Since ImagePipeline.process() does Load -> Preprocess -> Vision -> Risk -> Protect,
                    # we can't easily skip Vision+Risk using just `process()`.
                    # Let's manually do the Preprocess -> Protect flow for tracking frames.
                    from packages.image_processing.preprocessor import resize_preserve_aspect_ratio
                    processed_frame = resize_preserve_aspect_ratio(frame_rgb, self.config.image.max_size)
                    
                    protected_rgb = self.image_pipeline.protection.protect(processed_frame, analysis)
                    
                # Write out
                out_bgr = cv2.cvtColor(protected_rgb, cv2.COLOR_RGB2BGR)
                
                if writer is None:
                    # Initialize writer on the very first frame to get exact dimensions
                    out_h, out_w = out_bgr.shape[:2]
                    writer = create_writer(output_path, meta["fps"], out_w, out_h)
                    
                writer.write(out_bgr)
                frame_count += 1
                
                # Optional: logging progress
                if frame_count % (int(video_fps) * 5) == 0:  # log every 5 seconds of video
                    logger.debug(f"Processed {frame_count}/{meta['frame_count']} frames")
                    
            elapsed = time.time() - start_time
            log_event(
                logger, 
                logging.INFO, 
                LogEvent.VIDEO_ANALYSIS_COMPLETED, 
                f"Video processing completed in {elapsed:.3f}s",
                metadata={"elapsed_seconds": elapsed, "frames_processed": frame_count}
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            log_event(
                logger, 
                logging.ERROR, 
                LogEvent.PROCESSING_ERROR, 
                f"Video pipeline error: {str(e)}",
                metadata={"elapsed_seconds": elapsed}
            )
            raise
            
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
