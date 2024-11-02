import glob
import os
from pathlib import Path
from typing import Dict, List
from pydub import AudioSegment

from tqdm import tqdm


def merge_audios(
    segments: List[AudioSegment],
    expected_duration_in_seconds: float = 3,
) -> AudioSegment:
    merged_segment = AudioSegment.empty()
    for segment in segments:
        merged_segment += segment

    if len(merged_segment) < expected_duration_in_seconds * 1000:
        symmetric_silence_duration = (
            expected_duration_in_seconds * 1000 - len(merged_segment)
        ) // 2
        silence = AudioSegment.silent(symmetric_silence_duration)
        merged_segment = silence + merged_segment + silence

    return merged_segment


def merge_audios_by_speakers(
    input_path: str,
    output_path: str,
    expected_duration_in_seconds: float = 3,
):
    for dataset in ["train", "test", "validation"]:
        print(f"Merging audio files by speaker for {dataset} data...")
        for label in ["0", "1"]:
            output_directory = os.path.join(output_path, dataset, label)
            os.makedirs(output_directory, exist_ok=True)
            speaker_to_audio: Dict[str, List[str]] = {}

            for file in tqdm(
                glob.glob(
                    f"{input_path}/{dataset}/{label}/**_nohash_0.wav",
                    recursive=True,
                )
            ):
                speaker_id = Path(file).name.split("_")[1]
                speaker_to_audio[speaker_id] = speaker_to_audio.get(speaker_id, [])
                speaker_to_audio[speaker_id].append(file)

            for speaker_id, audio_files in tqdm(speaker_to_audio.items()):
                audio_segments = [AudioSegment.from_wav(file) for file in audio_files]
                merged_audio = merge_audios(
                    audio_segments,
                    expected_duration_in_seconds=expected_duration_in_seconds,
                )
                merged_audio.export(
                    f"{output_directory}/{speaker_id}.wav",
                    format="wav",
                )
