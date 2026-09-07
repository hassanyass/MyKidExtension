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
            
            # Actually, `ImagePipeline` returns `final_image`.
            # Let's peek at the first frame to see what resolution it gives us back.
            success, first_frame = cap.read()
            if not success:
                raise VideoProcessingError("Failed to read the first frame of the video")
                
            # Process first frame to establish output dimensions
            # We must convert BGR to RGB before passing to ImagePipeline, because ImagePipeline loader expects raw bytes/paths
            # BUT if we pass a numpy array, `load_image` will handle it (assuming it's BGR if it's 3-channels).
            # Wait, `load_image` currently assumes 3-channel numpy arrays are BGR and converts to RGB.
            # Let's just pass the BGR frame directly to `process`.
            protected_frame, _ = self.image_pipeline.process(first_frame)
            
            # The returned `protected_frame` is RGB. cv2.VideoWriter expects BGR.
            out_bgr = cv2.cvtColor(protected_frame, cv2.COLOR_RGB2BGR)
            out_h, out_w = out_bgr.shape[:2]
            
            # Now we know our output dimensions
            writer = create_writer(output_path, meta["fps"], out_w, out_h)
            writer.write(out_bgr)
            
            # 3. Process the rest of the frames
            frame_count = 1
            while True:
                success, frame = cap.read()
                if not success:
                    break
                    
                protected_frame, _ = self.image_pipeline.process(frame)
                out_bgr = cv2.cvtColor(protected_frame, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)
                frame_count += 1
                
                # Optional: logging progress
                if frame_count % 30 == 0:
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
