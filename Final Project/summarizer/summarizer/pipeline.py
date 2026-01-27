# pipeline.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_KERAS"] = "1"

import re
import torch
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
import whisper
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =====================
# LOAD MODELS (ONCE)
# =====================
print("⏳ Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("⏳ Loading sentence embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("⏳ Loading summarization model...")
summarizer = pipeline(
    task="summarization",                 # ✅ CORRECT TASK
    model="sshleifer/distilbart-cnn-12-6",
    device=-1                             # ✅ FORCE CPU (stable)
)


# =====================
# TEXT SUMMARIZER
# =====================
def summarize_text(text, max_chunk_words=400):
    if not text or not text.strip():
        return ""

    summaries = []
    words = text.split()

    for i in range(0, len(words), max_chunk_words):
        chunk_words = words[i:i + max_chunk_words]
        if len(chunk_words) < 40:
            continue

        chunk = " ".join(chunk_words)

        max_len = min(90, len(chunk_words))
        min_len = min(30, max_len - 1)
        if max_len <= min_len:
            continue

        try:
            result = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                truncation=True,
                do_sample=False,
            )
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            print("⚠ Summarization error:", e)
            continue

    return " ".join(summaries)


# =====================
# MAIN PIPELINE
# =====================
def run_pipeline(audio_path):
    print("▶ Starting pipeline for:", audio_path)

    # ---------- AUDIO ----------
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000).normalize()

    tmp_wav = "temp_pipeline.wav"
    audio.export(tmp_wav, format="wav")

    result = whisper_model.transcribe(tmp_wav)
    os.remove(tmp_wav)

    # ---------- CLEAN ----------
    def clean_text(t):
        t = t.lower()
        t = re.sub(r"\.{2,}", "", t)
        t = re.sub(r"[^a-z0-9\s?.!']", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    sentences = [
        clean_text(seg["text"])
        for seg in result["segments"]
        if seg["text"].strip()
    ]

    if len(sentences) < 10:
        print("⚠ Not enough speech detected")
        return []

    # ---------- WINDOWING ----------
    WINDOW = 7 if len(sentences) < 500 else 10
    windows = [
        " ".join(sentences[i:i + WINDOW])
        for i in range(len(sentences) - WINDOW + 1)
    ]

    embeddings = embedder.encode(
        windows,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # ---------- SPLITTING ----------
    THRESHOLD = 0.68
    sims = [
        cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        for i in range(len(embeddings) - 1)
    ]

    split_indices = [i + 1 for i, s in enumerate(sims) if s < THRESHOLD]

    topics, start = [], 0
    for idx in split_indices:
        topics.append(" ".join(sentences[start:idx]))
        start = idx
    topics.append(" ".join(sentences[start:]))

    # ---------- MERGE SMALL TOPICS ----------
    MIN_WORDS = 180
    MAX_WORDS = 900

    merged_topics = []
    current = ""

    for t in topics:
        curr_len = len(current.split())

        if curr_len < MIN_WORDS:
            current += " " + t
        elif curr_len > MAX_WORDS:
            merged_topics.append(current.strip())
            current = t
        else:
            merged_topics.append(current.strip())
            current = t

    if current.strip():
        merged_topics.append(current.strip())

    # ---------- LABEL + SUMMARY ----------
    output = []

    for i, t in enumerate(merged_topics):
        if i == 0:
            label = "INTRODUCTION"
        elif i == len(merged_topics) - 1:
            label = "CONCLUSION"
        else:
            label = f"MAIN PART {i}"

        summary = summarize_text(t)
        if summary:
            output.append((label, summary))

    print("✅ Pipeline finished")
    return output