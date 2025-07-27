# ✅ Final, Fast, Error-Free, Render-Compatible Speaker Extractor
# Version: Face + Voice + Noise Reduction + Render-Ready + Fast + Robust

import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import noisereduce as nr
import soundfile as sf
import numpy as np

CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")

# -----------------------------
# 1. Detect visible speaker (face)
# -----------------------------
def detect_speaker_segments(video_path: str, frame_skip=2, min_segment_dur=5.0, pad=0.2, resize_width=960) -> List[Tuple[float, float]]:
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    segments, active_start, frame_idx = [], None, 0

    while True:
        for _ in range(max(frame_skip - 1, 0)):
            if not cap.grab():
                break
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            frame = cv2.resize(frame, (resize_width, int(h * (resize_width / w))))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        t = frame_idx / fps
        if len(faces) > 0:
            if active_start is None:
                active_start = max(t - pad, 0)
        elif active_start is not None:
            end_t = t + pad
            if end_t - active_start >= min_segment_dur:
                segments.append((active_start, end_t))
            active_start = None
        frame_idx += frame_skip

    if active_start is not None:
        end_t = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        if end_t - active_start >= min_segment_dur:
            segments.append((active_start, end_t))
    cap.release()
    return segments

# -----------------------------
# 2. Detect speech intervals from audio
# -----------------------------
def detect_speech_intervals(audio_path, silence_thresh_db=-45, min_silence_len_ms=400, keep_silence_ms=100):
    audio = AudioSegment.from_file(audio_path)
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh_db)
    return [(max(0, s-keep_silence_ms)/1000.0, min(len(audio), e+keep_silence_ms)/1000.0) for s, e in nonsilent]

# -----------------------------
# 3. Intersect face and voice intervals
# -----------------------------
def intersect_intervals(a, b):
    res, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s, e = max(s1, s2), min(e1, e2)
        if e - s >= 5.0:
            res.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return res

# -----------------------------
# 4. Speaker segment extractor (with noise reduction)
# -----------------------------
def extract_speaker_clips(video_path, output_folder="output_clips", frame_skip=2, min_segment_dur=5.0,
                          threads=4, preset="ultrafast", silence_thresh_db=-45,
                          min_silence_len_ms=400, keep_silence_ms=100):
    os.makedirs(output_folder, exist_ok=True)
    face_segments = detect_speaker_segments(video_path, frame_skip, min_segment_dur)
    clip = VideoFileClip(video_path)
    temp_audio = os.path.join(output_folder, "_audio.wav")
    clip.audio.write_audiofile(temp_audio, logger=None)
    speech_segments = detect_speech_intervals(temp_audio, silence_thresh_db, min_silence_len_ms, keep_silence_ms)
    merged = intersect_intervals(face_segments, speech_segments)
    output_paths = []

    for i, (start, end) in enumerate(merged):
        if end - start < 5:
            continue
        subclip = clip.subclip(start, end)
        audio_array = subclip.audio.to_soundarray(fps=44100)
        if len(audio_array.shape) == 2:
            audio_array = np.mean(audio_array, axis=1)  # Mono
        reduced = nr.reduce_noise(y=audio_array, sr=44100)
        reduced_path = os.path.join(output_folder, f"temp_clean_audio_{i+1}.wav")
        sf.write(reduced_path, reduced, 44100)
        final_path = os.path.join(output_folder, f"speaker_clip_{i+1:03d}.mp4")
        subclip.set_audio(VideoFileClip(reduced_path).audio).write_videofile(
            final_path, codec="libx264", audio_codec="aac",
            preset=preset, threads=threads, logger=None
        )
        output_paths.append(final_path)

    zip_path = os.path.join(output_folder, "speaker_segments.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in output_paths:
            zf.write(p, arcname=os.path.basename(p))
    clip.close()
    return zip_path, f"Extracted {len(output_paths)} speaker clips."

# -----------------------------
# 5. Gradio + Render Launch Glue
# -----------------------------
def process_video(video, frame_skip, min_segment_dur, threads, preset, silence_thresh_db, min_silence_len_ms, keep_silence_ms):
    return extract_speaker_clips(video, frame_skip=frame_skip, min_segment_dur=min_segment_dur, threads=threads,
                                 preset=preset, silence_thresh_db=silence_thresh_db,
                                 min_silence_len_ms=min_silence_len_ms, keep_silence_ms=keep_silence_ms)

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🎤 Speaker Extractor (Face + Voice + Denoise)")
        video = gr.Video()
        with gr.Accordion("Advanced Settings", open=False):
            frame_skip = gr.Slider(1, 10, value=2)
            min_segment_dur = gr.Slider(5.0, 30.0, value=10.0)
            silence_thresh_db = gr.Slider(-80, -10, value=-45)
            min_silence_len_ms = gr.Slider(100, 2000, value=400)
            keep_silence_ms = gr.Slider(0, 1000, value=100)
            threads = gr.Slider(1, 8, value=4)
            preset = gr.Dropdown(["ultrafast", "superfast", "fast"], value="ultrafast")
        output = gr.File()
        status = gr.Textbox()
        btn = gr.Button("Process Video")
        btn.click(process_video, inputs=[video, frame_skip, min_segment_dur, threads, preset,
                                         silence_thresh_db, min_silence_len_ms, keep_silence_ms],
                  outputs=[output, status])
    return demo

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    build_ui().queue().launch(server_name="0.0.0.0", server_port=port)
