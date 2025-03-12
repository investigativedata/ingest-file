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
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
        segments, _ = model.transcribe(file_path, beam_size=5)

        for segment in segments:
            entity.add("bodyText", f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")