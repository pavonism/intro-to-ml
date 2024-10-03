import pyaudio
import wave
import threading


class AudioRecorder:
    def __init__(
        self,
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        chunk=1024,
    ):
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.CHUNK = chunk
        self.frames = []
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.thread = None

    def start_record(self):
        """Start recording audio."""
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        self.frames = []
        self.recording = True

        print("Recording started...")

        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        """Internal method to continuously record audio."""
        while self.recording:
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)

    def stop_record(self, output_filename: str):
        """Stop recording audio and save to the output file."""
        self.recording = False
        self.thread.join()

        self.stream.stop_stream()
        self.stream.close()

        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(self.frames))

        print(f"Recording stopped. Audio saved to {output_filename}")

    def is_microphone_available() -> bool:
        audio = pyaudio.PyAudio()

        mic_available = False
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)

            if device_info["maxInputChannels"] > 0:
                print(f"Microphone found: {device_info['name']}")
                mic_available = True

        audio.terminate()  # Clean up pyaudio resources
        return mic_available

    def __del__(self):
        """Clean up the audio resources."""
        self.audio.terminate()
