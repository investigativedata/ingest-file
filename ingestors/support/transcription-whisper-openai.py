import logging

import whisper

log = logging.getLogger(__name__)

MODEL_SIZE = "medium"

class TranscriptionSupport:
    """Provides a helper for transcribing audio and video files."""

    def transcribe(self, file_path, entity):
        """
        beam_size: https://stackoverflow.com/questions/22273119/what-does-the-beam-size-represent-in-the-beam-search-algorithm 
        """
        log.critical("loading model")
        model = whisper.load_model(MODEL_SIZE)
        log.critical(f"loading audio file from: {file_path}")
        audio = whisper.load_audio(file_path)
        log.critical("running pad_or_trim")
        audio = whisper.pad_or_trim(audio)
        log.critical("transcribing")
        result = model.transcribe(audio, verbose=True)

        # TODO chunking https://stackoverflow.com/a/57126101

        # can it return time stamps?
        entity.add("bodyText", result.text)