"""Movie Metadata RAG Assistant.

Flask + TF-IDF retrieval + Ollama (minimax-m2.1:cloud) for grounded
question answering over the movies_metadata.csv dataset.
"""

from __future__ import annotations

import os
import io
import math
import logging
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = "https://hiperc.buffalostate.edu/courses/movies_metadata.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "minimax-m2.1:cloud"
KEEP_COLS = ["title", "overview", "genres", "release_date", "vote_average"]
TOP_K_DEFAULT = 5
APP_PORT = 5005

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("movie_rag")


def load_movies() -> pd.DataFrame:
    """Load the movies CSV from the course URL and keep only useful columns."""
    log.info("Downloading dataset from %s", DATA_URL)
    # low_memory=False avoids dtype warnings on this mixed-type CSV
    df = pd.read_csv(DATA_URL, low_memory=False)

    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df[KEEP_COLS].copy()

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    df["genres"] = df["genres"].fillna("").astype(str)
    df["release_date"] = df["release_date"].fillna("").astype(str)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)

    # The original genres column is a stringified list of dicts. Pull the names.
    df["genres"] = df["genres"].apply(_extract_genre_names)

    df = df[(df["title"] != "") & (df["overview"] != "")].reset_index(drop=True)

    df["search_text"] = (
        df["title"] + " . " + df["genres"] + " . " + df["overview"]
    ).str.lower()

    log.info("Loaded %d movie rows after cleaning", len(df))
    return df


def _extract_genre_names(raw: str) -> str:
    """Convert the TMDB genres field (stringified list of dicts) to plain names."""
    if not isinstance(raw, str) or not raw or raw == "nan":
        return ""
    try:
        import ast

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            names = [str(item.get("name", "")) for item in parsed if isinstance(item, dict)]
            return ", ".join(n for n in names if n)
    except (ValueError, SyntaxError):
        pass
    return ""


def build_index(df: pd.DataFrame) -> Tuple[TfidfVectorizer, Any]:
    """Fit a TF-IDF vectorizer on the search_text column."""
    log.info("Building TF-IDF index over %d documents", len(df))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(df["search_text"].tolist())
    log.info("TF-IDF matrix shape: %s", matrix.shape)
    return vectorizer, matrix


def retrieve_movies(
    question: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """Return the top-k movies most similar to the question."""
    if not question or not question.strip():
        return []

    q_vec = vectorizer.transform([question.lower()])
    sims = cosine_similarity(q_vec, matrix).ravel()

    top_idx = sims.argsort()[::-1][:top_k]
    rows: List[Dict[str, Any]] = []
    for i in top_idx:
        score = float(sims[i])
        if score <= 0:
            continue
        r = df.iloc[int(i)]
        rows.append(
            {
                "title": r["title"],
                "overview": r["overview"],
                "genres": r["genres"],
                "release_date": r["release_date"],
                "vote_average": float(r["vote_average"]),
                "score": round(score, 4),
            }
        )
    return rows


def build_context(rows: List[Dict[str, Any]]) -> str:
    """Build a compact, numbered context block from retrieved rows."""
    if not rows:
        return "(no relevant movies were retrieved)"

    parts = []
    for i, r in enumerate(rows, start=1):
        overview = r["overview"]
        if len(overview) > 500:
            overview = overview[:500].rstrip() + "..."
        parts.append(
            f"[{i}] Title: {r['title']}\n"
            f"    Release: {r['release_date']} | Rating: {r['vote_average']} | Genres: {r['genres']}\n"
            f"    Overview: {overview}"
        )
    return "\n\n".join(parts)


def ask_ollama(question: str, context: str, timeout: int = 120) -> Tuple[bool, str]:
    """Send the grounded prompt to Ollama. Returns (ok, message)."""
    prompt = (
        "You are a careful movie assistant. Answer the user's question using ONLY "
        "the retrieved movie context below. If the context does not contain enough "
        "information to answer, reply exactly: \"The retrieved context does not "
        "contain enough information to answer that.\" Do not invent titles, dates, "
        "ratings, or facts that are not present in the context.\n\n"
        f"Retrieved Context:\n{context}\n\n"
        f"User Question: {question}\n\n"
        "Grounded Answer:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return False, (
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure Ollama is running (`ollama serve`) and the model "
            f"`{OLLAMA_MODEL}` is available."
        )
    except requests.exceptions.Timeout:
        return False, "Ollama request timed out. Try a shorter question or check the model."
    except requests.exceptions.RequestException as exc:
        return False, f"Ollama request failed: {exc}"

    if resp.status_code != 200:
        return False, f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        data = resp.json()
    except ValueError:
        return False, "Ollama returned a non-JSON response."

    answer = (data.get("response") or "").strip()
    if not answer:
        return False, "Ollama returned an empty response."
    return True, answer


app = Flask(__name__)

log.info("Initializing Movie Metadata RAG Assistant...")
MOVIES_DF = load_movies()
VECTORIZER, TFIDF_MATRIX = build_index(MOVIES_DF)
log.info("Ready. Open http://127.0.0.1:%d", APP_PORT)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    rows: List[Dict[str, Any]] = []
    answer = ""
    error = ""
    context_preview = ""

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()

        if not question:
            error = "Please enter a question before submitting."
        else:
            rows = retrieve_movies(question, MOVIES_DF, VECTORIZER, TFIDF_MATRIX, TOP_K_DEFAULT)
            context_preview = build_context(rows)

            if not rows:
                error = (
                    "No relevant movies were found in the dataset for that question. "
                    "Try rephrasing it."
                )
            else:
                ok, msg = ask_ollama(question, context_preview)
                if ok:
                    answer = msg
                else:
                    error = msg

    return render_template(
        "index.html",
        question=question,
        rows=rows,
        answer=answer,
        error=error,
        context_preview=context_preview,
        model=OLLAMA_MODEL,
        dataset_size=len(MOVIES_DF),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=APP_PORT, debug=False)
