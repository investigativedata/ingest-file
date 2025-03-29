import gc
import logging

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

MODEL_SIZE = "large-v3"

class TranscriptionSupport:
    """Provides a helper for transcribing audio and video files."""

    def transcribe(self, file_path, entity):
        """
        A description of the arguments for the WhisperModel init:
        https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py#L603
        beam_size: https://stackoverflow.com/questions/22273119/what-does-the-beam-size-represent-in-the-beam-search-algorithm 
        """

        self.model = None

        try:
            # compute_type="float32"
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
            log.info(f"Transcription model initialized successfully.")

            segments, _ = self.model.transcribe(file_path, vad_filter=True, beam_size=5, no_speech_threshold=0.6, condition_on_previous_text=False)

            for segment in segments:
                entity.add("bodyText", f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                log.info(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        finally:
            self._del_model()

    def _del_model(self):
        # https://github.com/SYSTRAN/faster-whisper/issues/660
        self.model.model.unload_model()
        
        if hasattr(self.model, 'model'):
            del self.model.model
        if hasattr(self.model, 'feature_extractor'):
            del self.model.feature_extractor
        if hasattr(self.model, 'hf_tokenizer'):
            del self.model.hf_tokenizer
        
        del self.model

        gc.collect()

        log.info("Transcription model removed from memory.")