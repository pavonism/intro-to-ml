import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog
from backend.audio_recorder import AudioRecorder
from backend.intercom_classifier import WordAudioClassifier


def run_app():
    """
    The entry point of the application.
    Opens a tkinter window with buttons to start and stop recording.
    """

    recorder = AudioRecorder()
    file_name = ""

    def wait_for_classification():
        global file_name
        result = WordAudioClassifier(
            class_labels=os.listdir("data/tsrc_spectrograms_cleaned/train"),
            model_path="models/robustness_3_single_cnn_tsrc_spectrograms_cleaned_augmented_sharpened/",
            verbose=True,
        ).predict_word(file_name)

        text_output.delete("1.0", tk.END)

        text_output.insert(tk.END, result)

    def check_recording():
        global file_name
        file_name = file_name_textbox.get("1.0", tk.END).strip()
        if not file_name:
            text_output.config(state=tk.NORMAL)
            text_output.delete("1.0", tk.END)
            text_output.insert(tk.END, "No file selected.", "red")
            text_output.tag_config("red", foreground="red")
            text_output.config(state=tk.DISABLED)
            return

        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, f"Checking audio file: {file_name}\n")

        waiting_thread = threading.Thread(target=wait_for_classification)
        waiting_thread.start()

    def stop_recording():
        global file_name
        temp_dir = tempfile.gettempdir()
        file_name = f"{temp_dir}/recording.wav"
        recorder.stop_record(file_name)
        file_name_textbox.delete("1.0", tk.END)
        file_name_textbox.insert(tk.END, file_name)

    def pick_file():
        global file_name
        file_name = filedialog.askopenfilename()
        file_name_textbox.delete("1.0", tk.END)
        file_name_textbox.insert(tk.END, file_name)

    app = tk.Tk()
    app.title("Audio intercom classifier")

    frame = tk.Frame(app)
    frame.pack(pady=10)

    # Top panel of the window
    top_frame = tk.Frame(frame)
    top_frame.pack(side=tk.TOP, padx=10)

    record_button = tk.Button(
        top_frame,
        text="Start Recording",
        command=recorder.start_record,
    )
    record_button.pack(side=tk.LEFT, padx=10)

    stop_button = tk.Button(
        top_frame,
        text="Stop Recording",
        command=stop_recording,
    )
    stop_button.pack(side=tk.LEFT, padx=10)

    file_button = tk.Button(
        top_frame,
        text="Select File",
        command=pick_file,
    )
    file_button.pack(side=tk.LEFT, padx=10)

    # Middle panel of the window
    middle_frame = tk.Frame(frame)
    middle_frame.pack(side=tk.TOP, padx=10)

    run_button = tk.Button(
        middle_frame,
        text="Check Recording",
        command=check_recording,
    )
    run_button.pack(side=tk.LEFT, padx=10)

    file_name_textbox = tk.Text(middle_frame, height=1, width=50)
    file_name_textbox.pack(side=tk.RIGHT, pady=10)
    file_name_textbox.insert(tk.END, file_name)

    # Bottom of the window
    bottom_frame = tk.Frame(frame)
    bottom_frame.pack(side=tk.TOP, padx=10)

    text_output = tk.Text(bottom_frame, height=10, width=50, font=("Courier", 16))
    text_output.pack(pady=10)
    text_output.config(state=tk.DISABLED)

    app.mainloop()
