import json
import glob
import logging
import subprocess
from pathlib import Path

from ingestors.exc import ProcessingException

log = logging.getLogger(__name__)

DESIRED_MODEL = "ggml-medium.en.bin"
DEFAULT_MODEL = "ggml-medium.en.bin"
TRANS_TIMEOUT = 60 * 60 # maximum seconds a transcription can run

class TranscriptionSupport:
    """Provides a helper for transcribing audio and video files."""

    def transcribe(self, file_path, entity):
        model = None

        models_path = Path("/whisper/models")
        if models_path / DESIRED_MODEL in models_path.glob("*"):
            model = DESIRED_MODEL
            log.info(f"The desired transcription model ({DESIRED_MODEL}) was found.")
        else:
            model = DEFAULT_MODEL
            log.error(f"The desired transcription model ({DESIRED_MODEL}) isn't installed. Using the default ({DEFAULT_MODEL}).")

        output_path = Path("/ingestors") / file_path.parts[-1].split(".")[0]

        cmd = ["/whisper/build/bin/whisper-cli",
                "-m",
                models_path / model,
                "-f",
                file_path,
                "-oj",
                # "True",
                "-of",
                output_path
                ]   

        subprocess.run(cmd, timeout=TRANS_TIMEOUT, check=True)
        # if the transcription succeeded, the output is written to a JSON
        output_path = output_path.with_suffix(".json")

        with open(output_path) as f:
            transcription_dict = json.loads(f.read())
            
        transcription_intervals = transcription_dict.get("transcription")
        if transcription_intervals:
            full_transcription = ""
            for interval in transcription_intervals:
                full_transcription += f"[{interval['timestamps']['from']} -> {interval['timestamps']['to']}] {interval['text'].strip()}"
                log.debug(f"[{interval['timestamps']['from']} -> {interval['timestamps']['to']}] {interval['text'].strip()}")
            entity.add("bodyText", full_transcription)
        else:
            raise ProcessingException(f"Transcription failed, no output in file {output_path}.")
