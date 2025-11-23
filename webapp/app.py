import os
import json
import csv
import math
from pathlib import Path
from flask import Flask, render_template, send_file, request, abort, url_for

app = Flask(__name__)
app.jinja_env.globals.update(range=range, max=max, min=min)

RESULTS_DIR = os.path.expanduser("~/autodl-tmp/results")
PER_PAGE = 30

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def get_methods():
    if not os.path.isdir(RESULTS_DIR):
        return []
    methods = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        full = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(full) and not name.startswith("."):
            methods.append(name)
    return methods


def get_runs(method):
    runs = []
    method_dir = os.path.join(RESULTS_DIR, method)
    if not os.path.isdir(method_dir):
        return runs
    for name in sorted(os.listdir(method_dir)):
        full = os.path.join(method_dir, name)
        if os.path.isdir(full) and not name.startswith("."):
            runs.append(name)
    return runs


def get_audio_files(run_path):
    files = []
    for f in sorted(os.listdir(run_path)):
        if not f.startswith("_") and not f.startswith("."):
            ext = os.path.splitext(f)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                files.append(f)
    return files


def load_metrics(run_path):
    metrics_json_path = os.path.join(run_path, "metrics.json")
    metrics_csv_path = os.path.join(run_path, "_results.csv")

    result = {"has_metrics": False, "summary": None, "columns": None, "rows": None}

    if os.path.isfile(metrics_json_path):
        try:
            with open(metrics_json_path) as f:
                data = json.load(f)
            result["summary"] = data
            result["has_metrics"] = True
        except Exception:
            pass

    if os.path.isfile(metrics_csv_path):
        try:
            with open(metrics_csv_path, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if rows:
                result["columns"] = rows[0]
                result["rows"] = rows[1:]
        except Exception:
            pass

    return result


@app.route("/")
def index():
    methods = get_methods()
    return render_template("index.html", methods=methods)


@app.route("/method/<method>")
def method_view(method):
    runs = get_runs(method)
    if not runs:
        abort(404)
    return render_template("index.html", methods=get_methods(), selected_method=method, runs=runs)


@app.route("/run/<path:run_path>")
def run_view(run_path):
    full_path = os.path.join(RESULTS_DIR, run_path)
    if not os.path.isdir(full_path):
        abort(404)

    parts = run_path.split("/")
    method = parts[0] if parts else ""
    run_name = parts[1] if len(parts) > 1 else ""

    audio_files = get_audio_files(full_path)
    total = len(audio_files)

    page = request.args.get("page", 1, type=int)
    total_pages = max(1, math.ceil(total / PER_PAGE))
    page = max(1, min(page, total_pages))

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    page_files = audio_files[start:end]

    metrics = load_metrics(full_path)

    return render_template(
        "run.html",
        method=method,
        run_name=run_name,
        run_path=run_path,
        full_path=full_path,
        audio_files=page_files,
        total=total,
        page=page,
        total_pages=total_pages,
        per_page=PER_PAGE,
        metrics=metrics,
    )


@app.route("/audio/<path:audio_path>")
def serve_audio(audio_path):
    full_path = os.path.join(RESULTS_DIR, audio_path)
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path, mimetype="audio/wav")


@app.route("/api/metrics/<path:run_path>")
def api_metrics(run_path):
    full_path = os.path.join(RESULTS_DIR, run_path)
    if not os.path.isdir(full_path):
        abort(404)
    return load_metrics(full_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6008, debug=False)
