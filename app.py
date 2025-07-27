import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import soundfile as sf
import noisereduce as nr

from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# -----------------------------
# Globals
# -----------------------------
CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")


# -----------------------------
# Face-driven segment detector
# -----------------------------
def detect_speaker_segments(
    video_path: str,
    frame_skip: int = 2,
    min_segment_dur: float = 5.0,
    pad: float = 0.2,
    resize_width: int = 960,
) -> List[Tuple[float, float]]:
    """Return (start, end) segments where at least one face is visible long enough."""
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError("Failed to load haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    active_start = None
    segments: List[Tuple[float, float]] = []
    frame_idx = 0

    while True:
        for _ in range(max(frame_skip - 1, 0)):
            if not cap.grab():
                break

        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40),
        )

        t = frame_idx / fps
        if len(faces) > 0:
            if active_start is None:
                active_start = max(t - pad, 0)
        else:
            if active_start is not None:
                end_t = t + pad
                if end_t - active_start >= min_segment_dur:
                    segments.append((active_start, end_t))
                active_start = None

        frame_idx += frame_skip

    # flush last
    if active_start is not None:
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (fps or 1)
        end_t = duration
        if end_t - active_start >= min_segment_dur:
            segments.append((active_start, end_t))

    cap.release()

    # merge small gaps
    merged: List[Tuple[float, float]] = []
    merge_gap = 1.0
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        ps, pe = merged[-1]
        s, e = seg
        if s - pe <= merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append(seg)

    return merged


# -----------------------------
# Audio speech (silence removal) + denoise
# -----------------------------
def denoise_wav(in_path: str, out_path: str):
    data, sr = sf.read(in_path)
    if data.ndim == 2:
        # stereo -> mono
        data = data.mean(axis=1)
    reduced = nr.reduce_noise(y=data, sr=sr)
    sf.write(out_path, reduced, sr)


def detect_speech_intervals(
    audio_path: str,
    silence_thresh_db: int = -35,
    min_silence_len_ms: int = 800,
    keep_silence_ms: int = 100,
) -> List[Tuple[float, float]]:
    """Return list of (start, end) seconds where speech is present using pydub after denoise."""
    # Denoise first
    denoised = audio_path.replace(".wav", "_denoised.wav")
    denoise_wav(audio_path, denoised)

    audio = AudioSegment.from_file(denoised)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )

    expanded = []
    for start_ms, end_ms in ranges:
        start_ms = max(0, start_ms - keep_silence_ms)
        end_ms = min(len(audio), end_ms + keep_silence_ms)
        expanded.append((start_ms / 1000.0, end_ms / 1000.0))
    return expanded


# -----------------------------
# Interval utils
# -----------------------------
def intersect_intervals(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Return intersection between two sorted lists of [start, end] (seconds)."""
    res = []
    i = j = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s:
            res.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return res


def enforce_target_lengths(
    segs: List[Tuple[float, float]], target_min=20.0, target_max=30.0, max_merge_gap=5.0
) -> List[Tuple[float, float]]:
    """Merge/split segments to be ~target_min..target_max seconds."""
    if not segs:
        return []

    # merge with small gaps first
    merged = []
    cs, ce = segs[0]
    for s, e in segs[1:]:
        if s - ce <= max_merge_gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    # pack into target windows
    packed = []
    for s, e in merged:
        duration = e - s
        if duration < target_min:
            # skip tiny segments
            continue
        while duration > target_max:
            packed.append((s, s + target_max))
            s += target_max
            duration = e - s
        if duration >= target_min:
            packed.append((s, e))
    return packed


# -----------------------------
# Main extraction
# -----------------------------
def extract_speaker_clips(
    video_path: str,
    output_folder: str = "output_clips",
    frame_skip: int = 2,
    min_segment_dur: float = 5.0,
    threads: int = 4,
    preset: str = "ultrafast",
    silence_thresh_db: int = -35,
    min_silence_len_ms: int = 800,
    keep_silence_ms: int = 100,
):
    os.makedirs(output_folder, exist_ok=True)

    # 1) Face-based segments
    face_segments = detect_speaker_segments(
        video_path, frame_skip=frame_skip, min_segment_dur=min_segment_dur
    )
    if not face_segments:
        return None, "No speaker (face) segments detected."

    # 2) Whole audio (denoise + VAD)
    video_clip = VideoFileClip(video_path)
    raw_audio = os.path.join(output_folder, "_full_audio.wav")
    video_clip.audio.write_audiofile(raw_audio, logger=None)
    speech_segments = detect_speech_intervals(
        raw_audio,
        silence_thresh_db=silence_thresh_db,
        min_silence_len_ms=min_silence_len_ms,
        keep_silence_ms=keep_silence_ms,
    )

    # 3) Intersection
    merged_segments = intersect_intervals(face_segments, speech_segments)

    # 4) Shape to 20–30s
    merged_segments = enforce_target_lengths(
        merged_segments, target_min=20.0, target_max=30.0, max_merge_gap=5.0
    )

    if not merged_segments:
        video_clip.close()
        return None, "No overlapping (visible + speaking) segments of >=20s found."

    # 5) Export clips
    output_paths = []
    for i, (start, end) in enumerate(merged_segments, 1):
        sub = video_clip.subclip(start, end)
        out_path = os.path.join(output_folder, f"speaker_clean_{i:03d}.mp4")
        ffmpeg_params = ["-preset", preset, "-threads", str(threads)]
        sub.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=ffmpeg_params,
            logger=None,
        )
        output_paths.append(out_path)

    video_clip.close()

    # 6) Zip them
    zip_path = os.path.join(output_folder, "speaker_segments.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for path in output_paths:
            zipf.write(path, arcname=os.path.basename(path))

    return zip_path, f"✅ Extracted {len(output_paths)} clean 20–30s speaker clips."


# -----------------------------
# Gradio glue
# -----------------------------
def process_video(
    input_video,
    frame_skip: int,
    min_segment_dur: float,
    threads: int,
    preset: str,
    silence_thresh_db: int,
    min_silence_len_ms: int,
    keep_silence_ms: int,
):
    if not input_video:
        return None, "Please upload a video first."

    try:
        return extract_speaker_clips(
            input_video,
            frame_skip=frame_skip,
            min_segment_dur=min_segment_dur,
            threads=threads,
            preset=preset,
            silence_thresh_db=silence_thresh_db,
            min_silence_len_ms=min_silence_len_ms,
            keep_silence_ms=keep_silence_ms,
        )
    except Exception as e:
        return None, f"❌ Error: {e}"


def build_ui():
    with gr.Blocks(title="Speaker Extractor") as demo:
        gr.Markdown(
            """
# 🎤 Speaker Extractor (Face + Voice + Denoise)
**Keeps only segments where the speaker is visible and talking, denoises audio, and returns 20–30s clips.**
            """
        )

        video_input = gr.Video(label="Upload Video")
        with gr.Accordion("Advanced Settings", open=False):
            frame_skip = gr.Slider(1, 20, value=2, step=1, label="Frame Skip (higher = faster, less accurate)")
            min_segment_dur = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Min Segment Duration (for candidate segments, seconds)")
            silence_thresh_db = gr.Slider(-80, -10, value=-35, step=1, label="Silence Threshold (dBFS, higher = stricter)")
            min_silence_len_ms = gr.Slider(100, 2000, value=800, step=50, label="Min Silence Length (ms)")
            keep_silence_ms = gr.Slider(0, 1000, value=100, step=25, label="Silence Padding (ms)")
            threads = gr.Slider(1, 8, value=4, step=1, label="FFmpeg Threads")
            preset = gr.Dropdown(
                ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
                value="ultrafast",
                label="x264 Preset",
            )

        output_file = gr.File(label="📦 Download Speaker Clips (ZIP)")
        status_box = gr.Textbox(label="Status", interactive=False)

        gr.Button("🚀 Process").click(
            fn=process_video,
            inputs=[
                video_input,
                frame_skip,
                min_segment_dur,
                threads,
                preset,
                silence_thresh_db,
                min_silence_len_ms,
                keep_silence_ms,
            ],
            outputs=[output_file, status_box],
        )

    return demo


if __name__ == "__main__":
    # Render (Docker) — bind to 0.0.0.0 and a fixed port
    port = int(os.environ.get("PORT", 10000))
    build_ui().queue(concurrency_count=1, max_size=2).launch(
        server_name="0.0.0.0", server_port=port, show_error=True
    )
