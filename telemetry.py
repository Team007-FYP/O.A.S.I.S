# =============================================================================
# telemetry.py
# Part 6 — Benchmarking Pipeline: Hardware Spy, JSON Schema Builder,
#           and Markdown Master Report Generator.
#
# Roles:
#   HardwareSpy       -> Background thread that tracks peak VRAM usage via NVML.
#   JSONSchemaBuilder -> Produces a standardised result skeleton for every run.
#   ReportGenerator   -> Aggregates all JSON results into a ranked Markdown report.
# =============================================================================

import os           # Directory / path utilities
import json         # Serialise and deserialise result files
import time         # sleep() in the monitor loop
import threading    # Daemon thread for background VRAM polling
import datetime     # ISO-8601 timestamp generation
import glob         # Wildcard scanning of evaluation JSON files

import psutil       # CPU / system memory introspection (available for future extension)
import pynvml       # NVIDIA Management Library Python bindings


# =============================================================================
# 1. HARDWARE SPY
# Polls GPU VRAM in a background daemon thread every 200 ms and records the
# peak allocation observed during a model's inference window.
# =============================================================================

class HardwareSpy:
    """
    Background VRAM tracker.

    Usage pattern:
        spy = HardwareSpy()
        spy.start()
        # ... run model inference ...
        peak_mb = spy.stop()
    """

    def __init__(self) -> None:
        # Initialise the NVML library — must be called before any NVML query.
        # Raises pynvml.NVMLError if no NVIDIA driver is present.
        pynvml.nvmlInit()

        # Peak VRAM seen so far (in MB); updated by the monitor thread.
        self.peak_vram_mb: float = 0.0

        # Guard flag read/written by both the main thread (stop) and the monitor thread.
        self.is_tracking: bool = False

        # Mutex protecting peak_vram_mb from concurrent read-write races.
        self._lock: threading.Lock = threading.Lock()

        # Placeholder for the background thread object; set in start().
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------------------
    def _monitor_loop(self) -> None:
        """
        Inner polling loop executed on the daemon thread.
        Queries VRAM every 200 ms and updates self.peak_vram_mb.
        Terminates when self.is_tracking flips to False (set by stop()).
        """
        # Obtain a handle to GPU device 0 (the primary A100).
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        while self.is_tracking:
            # nvmlDeviceGetMemoryInfo returns an object with .used in bytes.
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_vram_mb: float = mem_info.used / (1024 ** 2)   # bytes -> MB

            # Update peak under the lock to prevent a torn read in stop().
            with self._lock:
                if current_vram_mb > self.peak_vram_mb:
                    self.peak_vram_mb = current_vram_mb

            # Poll interval: 200 ms — fine-grained enough to catch short spikes.
            time.sleep(0.2)

    # -------------------------------------------------------------------------
    def start(self) -> None:
        """
        Arm the spy and launch the background monitor thread.
        The thread is a daemon so it will not block interpreter shutdown.
        """
        self.is_tracking = True

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="HardwareSpy-VRAMMonitor",
            daemon=True,          # Thread dies automatically if main process exits
        )
        self._thread.start()

    # -------------------------------------------------------------------------
    def stop(self) -> float:
        """
        Disarm the spy, join the monitor thread, shut down NVML, and return
        the exact peak VRAM observed (in MB) during the tracked window.

        Returns
        -------
        float
            Peak GPU VRAM allocation in megabytes.
        """
        # Signal the loop to exit on its next iteration check.
        self.is_tracking = False

        # Wait for the monitor thread to finish its current sleep and exit cleanly.
        if self._thread is not None:
            self._thread.join(timeout=2.0)   # 2 s grace period beyond the 0.2 s poll

        # Release the NVML library resources.
        pynvml.nvmlShutdown()

        # Return the peak VRAM recorded; read under lock for safety.
        with self._lock:
            return self.peak_vram_mb


# =============================================================================
# 2. JSON SCHEMA BUILDER
# Produces a fully-typed, standardised result skeleton for every
# (meeting, model) pair.  Null fields are filled in by later pipeline stages.
# =============================================================================

class JSONSchemaBuilder:
    """
    Factory for result JSON skeletons.

    All null fields are intended to be populated by downstream pipeline stages
    (transcription, OCR, inference, evaluation) before the file is written to disk.
    """

    @staticmethod
    def build_skeleton(meeting_id: str, model_name: str) -> dict:
        """
        Construct and return a standardised result dictionary.

        Parameters
        ----------
        meeting_id : str
            Unique identifier for the meeting being processed (e.g. "M1").
        model_name : str
            Short pipeline key for the LLM being benchmarked (e.g. "glm4").

        Returns
        -------
        dict
            A fully-typed skeleton with all fields initialised to null / defaults.
        """
        return {
            # ------------------------------------------------------------------
            # TOP-LEVEL EXPERIMENT METADATA
            # ------------------------------------------------------------------
            "experiment": meeting_id,               # Which meeting this result covers
            "dataset": "AMI Corpus",                # Fixed dataset label for this pipeline
            "temperature": 0.0,                     # Must match InferenceConfig.TEMPERATURE
            "timestamp": datetime.datetime.now(     # ISO-8601 wall-clock at result creation
                datetime.timezone.utc
            ).isoformat(),

            # ------------------------------------------------------------------
            # MODEL IDENTITY
            # Quantization, context length, and parameter count are filled in
            # dynamically by the loader stage once the model card is inspected.
            # ------------------------------------------------------------------
            "model": {
                "name": model_name,
                "parameter_size": None,             # e.g. "9B", "8B", "7B"
                "quantization": None,               # e.g. "bfloat16", "int4"
                "context_length": None,             # Max tokens the model supports
            },

            # ------------------------------------------------------------------
            # INPUT METADATA
            # Populated after the audio/PDF processing stages complete.
            # ------------------------------------------------------------------
            "input": {
                "meeting_id": meeting_id,
                "audio_duration_sec": None,         # Length of A1.wav in seconds
                "pdf_pages": None,                  # Number of pages in P1.pdf
                "num_speakers": None,               # Diarisation speaker count
                "input_tokens": None,               # Prompt token count fed to the LLM
                "output_tokens": None,              # Generated token count
                "input_mode": "multimodal",         # Fixed: audio transcript + slide OCR
            },

            # ------------------------------------------------------------------
            # EFFICIENCY METRICS
            # Populated by the inference harness after generation completes.
            # ------------------------------------------------------------------
            "efficiency": {
                "inference_time_sec": None,         # Wall time for model.generate()
                "latency_sec": None,                # Time-to-first-token (TTFT)
                "throughput_tokens_per_sec": None,  # output_tokens / inference_time_sec
                "peak_vram_mb": None,               # Peak VRAM from HardwareSpy.stop()
            },

            # ------------------------------------------------------------------
            # SUMMARY QUALITY METRICS
            # Populated by the evaluation stage (ROUGE, BERTScore, BARTScore).
            # ------------------------------------------------------------------
            "summary_evaluation": {
                "rouge": {
                    "rouge1": {"precision": None, "recall": None, "f1": None},
                    "rouge2": {"precision": None, "recall": None, "f1": None},
                    "rougeL": {"precision": None, "recall": None, "f1": None},
                },
                "bertscore": {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "model": None,                  # e.g. "microsoft/deberta-xlarge-mnli"
                },
                "bartscore": {
                    "score": None,                  # Log probability (higher is better)
                    "model": None,                  # e.g. "facebook/bart-large-cnn"
                },
            },

            # ------------------------------------------------------------------
            # HALLUCINATION METRICS
            # Placeholder until a dedicated factual-consistency model is integrated.
            # ------------------------------------------------------------------
            "hallucination": {
                "factual_consistency_score": None,  # 0.0–1.0; model TBD
                "entity_hallucination_rate": None,  # % of entities not grounded in context
                "notes": (
                    "Placeholder metrics pending dedicated factual-consistency "
                    "model integration."
                ),
            },

            # ------------------------------------------------------------------
            # LLM-AS-JUDGE SCORE
            # Populated by the judge stage (Meta-Llama-3.1-8B-Instruct evaluator).
            # Typically a 0–10 scalar rating of overall summary quality.
            # ------------------------------------------------------------------
            "judge_score": None,
        }


# =============================================================================
# 3. REPORT GENERATOR
# Scans the evaluations directory, parses all JSON result files, computes a
# composite ranking score, and writes a formatted Markdown master report.
# =============================================================================

# =============================================================================
# 3. REPORT GENERATOR (Interactive Standalone HTML Dashboard + Plotly)
# =============================================================================

class ReportGenerator:
    """
    Aggregates per-model JSON results across all evaluation runs, computes global
    mean metrics, calculates normalized composite scores, and writes a standalone
    interactive HTML dashboard (master_report.html) styled with Tailwind CSS and
    Plotly.js.
    """

    @staticmethod
    def _safe_get(d: dict, *keys, default=None):
        """Traverse a nested dict safely; return default if any key is missing."""
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key, default)
        return d

    @staticmethod
    def _mean(values: list[float | None]) -> float:
        """Calculate arithmetic mean, safely filtering out None values."""
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    @staticmethod
    def _normalise(values: list[float]) -> list[float]:
        """Min-max normalise a list of values to a 0.0 - 1.0 range."""
        if not values or max(values) == min(values):
            return [0.5 for _ in values]
        lo, hi = min(values), max(values)
        return [round((v - lo) / (hi - lo), 4) for v in values]

    def generate_html(self, evals_dir: str, output_path: str = "master_report.html") -> str:
        """
        Parse all JSON evaluation files, aggregate metrics by model, and render
        a self-contained, interactive HTML dashboard using Plotly and Tailwind CSS.
        """
        pattern = os.path.join(evals_dir, "*.json")
        json_files = sorted(glob.glob(pattern))

        if not json_files:
            print(f"[ReportGenerator] No JSON files found in '{evals_dir}'. Aborting.")
            return ""

        # ------------------------------------------------------------------
        # 1. PARSE & GROUP METRICS BY MODEL
        # ------------------------------------------------------------------
        model_stats: dict[str, dict] = {}
        unique_meetings = set()

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:
                print(f"[ReportGenerator] Skipping corrupt file '{filepath}': {exc}")
                continue

            model_name = self._safe_get(data, "model", "name", default="unknown")
            meeting_id = self._safe_get(data, "experiment", default="unknown")
            unique_meetings.add(meeting_id)

            if model_name not in model_stats:
                model_stats[model_name] = {
                    "rouge_l": [],
                    "bert": [],
                    "judge": [],
                    "vram": [],
                    "throughput": [],
                    "runs": 0,
                }

            model_stats[model_name]["runs"] += 1
            model_stats[model_name]["rouge_l"].append(
                self._safe_get(data, "summary_evaluation", "rouge", "rougeL", "f1")
            )
            model_stats[model_name]["bert"].append(
                self._safe_get(data, "summary_evaluation", "bertscore", "f1")
            )
            model_stats[model_name]["judge"].append(
                self._safe_get(data, "judge_score")
            )
            model_stats[model_name]["vram"].append(
                self._safe_get(data, "efficiency", "peak_vram_mb")
            )
            model_stats[model_name]["throughput"].append(
                self._safe_get(data, "efficiency", "throughput_tokens_per_sec")
            )

        if not model_stats:
            print("[ReportGenerator] All JSON files failed to parse. Aborting.")
            return ""

        # ------------------------------------------------------------------
        # 2. COMPUTE GLOBAL AVERAGES
        # ------------------------------------------------------------------
        rows = []
        for model_name, s in model_stats.items():
            rows.append({
                "model": model_name,
                "runs": s["runs"],
                "avg_rouge_l": self._mean(s["rouge_l"]),
                "avg_bert": self._mean(s["bert"]),
                "avg_judge": self._mean(s["judge"]),
                "avg_vram": self._mean(s["vram"]),
                "avg_throughput": self._mean(s["throughput"]),
            })

        # ------------------------------------------------------------------
        # 3. NORMALIZATION & COMPOSITE SCORING (0.0 - 3.0)
        # ------------------------------------------------------------------
        norm_rouge = self._normalise([r["avg_rouge_l"] for r in rows])
        norm_bert = self._normalise([r["avg_bert"] for r in rows])
        norm_judge = self._normalise([r["avg_judge"] for r in rows])

        for i, row in enumerate(rows):
            row["norm_rouge"] = norm_rouge[i]
            row["norm_bert"] = norm_bert[i]
            row["norm_judge"] = norm_judge[i]
            row["composite"] = round(norm_rouge[i] + norm_bert[i] + norm_judge[i], 4)

        # Rank models descending by composite score
        rows.sort(key=lambda r: r["composite"], reverse=True)
        winner = rows[0]

        # ------------------------------------------------------------------
        # 4. PREPARE PLOTLY JSON DATA
        # ------------------------------------------------------------------
        radar_categories = ["ROUGE-L", "BERTScore", "Judge Quality", "Throughput"]
        norm_thr = self._normalise([r["avg_throughput"] for r in rows])

        radar_traces = []
        colors = ["#38bdf8", "#818cf8", "#34d399", "#f472b6", "#fbbf24"]

        for idx, r in enumerate(rows):
            r_vals = [
                r["norm_rouge"],
                r["norm_bert"],
                r["norm_judge"],
                norm_thr[idx],
                r["norm_rouge"],  # Close the radar loop
            ]
            radar_traces.append({
                "type": "scatterpolar",
                "r": r_vals,
                "theta": radar_categories + [radar_categories[0]],
                "fill": "toself",
                "name": r["model"],
                "line": {"color": colors[idx % len(colors)]},
                "opacity": 0.7,
            })

        scatter_traces = [{
            "x": [r["avg_throughput"] for r in rows],
            "y": [r["avg_judge"] for r in rows],
            "text": [f"<b>{r['model']}</b><br>Composite: {r['composite']}" for r in rows],
            "mode": "markers+text",
            "textposition": "top center",
            "marker": {
                "size": [max(18, min(40, r["composite"] * 15)) for r in rows],
                "color": colors[:len(rows)],
                "line": {"color": "#ffffff", "width": 1.5},
            },
            "type": "scatter",
        }]

        # ------------------------------------------------------------------
        # 5. HTML TEMPLATE INJECTION
        # ------------------------------------------------------------------
        report_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        table_rows_html = ""
        for rank, r in enumerate(rows, start=1):
            badge = {1: "🥇 1", 2: "🥈 2", 3: "🥉 3"}.get(rank, f"#{rank}")
            is_winner = "border-emerald-500/30 bg-emerald-950/20" if rank == 1 else "border-slate-800 bg-slate-900/40"
            table_rows_html += f"""
            <tr class="border-b {is_winner} hover:bg-slate-800/50 transition">
                <td class="px-6 py-4 font-bold text-slate-200">{badge}</td>
                <td class="px-6 py-4 font-mono font-bold text-sky-400">{r['model']}</td>
                <td class="px-6 py-4 text-center text-slate-300">{r['runs']}</td>
                <td class="px-6 py-4 text-right text-slate-200 font-mono">{r['avg_rouge_l']:.4f}</td>
                <td class="px-6 py-4 text-right text-slate-200 font-mono">{r['avg_bert']:.4f}</td>
                <td class="px-6 py-4 text-right text-amber-400 font-mono font-bold">{r['avg_judge']:.2f}</td>
                <td class="px-6 py-4 text-right text-slate-300 font-mono">{r['avg_vram']:.1f} MB</td>
                <td class="px-6 py-4 text-right text-emerald-400 font-mono">{r['avg_throughput']:.2f}</td>
                <td class="px-6 py-4 text-right font-mono font-extrabold text-cyan-300 text-base">{r['composite']:.4f}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal AI Benchmark Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        brand: {{ 50: '#f0fdf4', 500: '#22c55e', 900: '#14532d' }}
                    }}
                }}
            }}
        }}
    </script>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased min-h-screen p-6 md:p-12">
    <div class="max-w-7xl mx-auto space-y-8">

        <!-- HEADER -->
        <div class="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-6 gap-4">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
                    📊 Multimodal Summarisation Benchmark
                </h1>
                <p class="text-sm text-slate-400 mt-1">Cross-Modal Evaluation: WhisperX Diarization + Moondream2 VLM + Multi-LLM Fusion</p>
            </div>
            <div class="text-right">
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950/80 border border-emerald-500/40 text-emerald-400">
                    ● A100 GPU Complete Run
                </span>
                <p class="text-xs text-slate-500 mt-1 font-mono">{report_ts}</p>
            </div>
        </div>

        <!-- KPI CARDS -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="p-5 rounded-xl border border-emerald-500/30 bg-gradient-to-br from-emerald-950/30 to-slate-900/70 shadow-lg">
                <p class="text-xs font-bold uppercase tracking-wider text-emerald-400">Overall Winner</p>
                <h3 class="text-2xl font-black text-white mt-1 font-mono">{winner['model']}</h3>
                <p class="text-xs text-slate-400 mt-1">Composite: <span class="text-emerald-400 font-bold">{winner['composite']}</span> / 3.0</p>
            </div>
            <div class="p-5 rounded-xl border border-slate-800 bg-slate-900/60 shadow-lg">
                <p class="text-xs font-bold uppercase tracking-wider text-slate-400">Datasets Processed</p>
                <h3 class="text-2xl font-black text-white mt-1">{len(unique_meetings)}</h3>
                <p class="text-xs text-slate-400 mt-1">{len(json_files)} total evaluations ingested</p>
            </div>
            <div class="p-5 rounded-xl border border-slate-800 bg-slate-900/60 shadow-lg">
                <p class="text-xs font-bold uppercase tracking-wider text-slate-400">Top Judge Rating</p>
                <h3 class="text-2xl font-black text-amber-400 mt-1 font-mono">{winner['avg_judge']:.1f}<span class="text-xs text-slate-500"> / 10</span></h3>
                <p class="text-xs text-slate-400 mt-1">Evaluated by Llama-3.1-8B</p>
            </div>
            <div class="p-5 rounded-xl border border-slate-800 bg-slate-900/60 shadow-lg">
                <p class="text-xs font-bold uppercase tracking-wider text-slate-400">Peak Model Throughput</p>
                <h3 class="text-2xl font-black text-sky-400 mt-1 font-mono">{max(r['avg_throughput'] for r in rows):.1f} <span class="text-xs text-slate-500">tok/s</span></h3>
                <p class="text-xs text-slate-400 mt-1">FlashAttention-2 Accelerated</p>
            </div>
        </div>

        <!-- LEADERBOARD TABLE -->
        <div class="border border-slate-800 rounded-xl overflow-hidden bg-slate-900/40 shadow-xl backdrop-blur-sm">
            <div class="p-5 border-b border-slate-800 flex items-center justify-between">
                <div>
                    <h2 class="text-lg font-bold text-white">Aggregated Performance Leaderboard</h2>
                    <p class="text-xs text-slate-400">Sorted by Global Composite Score (normalized sum of ROUGE-L, BERTScore, and LLM-as-a-Judge)</p>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm text-slate-300" id="metricsTable">
                    <thead class="bg-slate-900/90 text-xs uppercase font-semibold text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="px-6 py-3 cursor-pointer hover:text-white" onclick="sortTable(0)">Rank ⇕</th>
                            <th class="px-6 py-3 cursor-pointer hover:text-white" onclick="sortTable(1)">Model ⇕</th>
                            <th class="px-6 py-3 text-center cursor-pointer hover:text-white" onclick="sortTable(2)">Runs ⇕</th>
                            <th class="px-6 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable(3)">Avg ROUGE-L ⇕</th>
                            <th class="px-6 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable(4)">Avg BERTScore ⇕</th>
                            <th class="px-6 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable(5)">Avg Judge ⇕</th>
                            <th class="px-6 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable(6)">Peak VRAM ⇕</th>
                            <th class="px-6 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable(7)">Throughput ⇕</th>
                            <th class="px-6 py-3 text-right cursor-pointer hover:text-white" onclick="sortTable(8)">Composite ⇕</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- CHARTS SECTION -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- RADAR CHART -->
            <div class="p-5 border border-slate-800 rounded-xl bg-slate-900/40 shadow-xl">
                <h3 class="text-sm font-bold uppercase tracking-wider text-slate-300 mb-1">Architecture Trade-off Radar</h3>
                <p class="text-xs text-slate-500 mb-4">Normalized multi-axis balance across lexical, semantic, quality, and speed</p>
                <div id="radarChart" class="w-full h-80"></div>
            </div>

            <!-- SCATTER PLOT -->
            <div class="p-5 border border-slate-800 rounded-xl bg-slate-900/40 shadow-xl">
                <h3 class="text-sm font-bold uppercase tracking-wider text-slate-300 mb-1">Quality vs. Latency Pareto Frontier</h3>
                <p class="text-xs text-slate-500 mb-4">Throughput (tok/s) vs. Judge Score (1-10 scale)</p>
                <div id="scatterChart" class="w-full h-80"></div>
            </div>
        </div>

    </div>

    <!-- PLOTLY RENDER SCRIPT -->
    <script>
        const radarTraces = {json.dumps(radar_traces)};
        const radarLayout = {{
            polar: {{
                radialaxis: {{ visible: true, range: [0, 1], color: '#64748b' }},
                angularaxis: {{ color: '#94a3b8' }},
                bgcolor: 'rgba(15, 23, 42, 0.4)'
            }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8', family: 'sans-serif' }},
            margin: {{ t: 30, b: 30, l: 40, r: 40 }},
            legend: {{ orientation: 'h', y: -0.15 }}
        }};
        Plotly.newPlot('radarChart', radarTraces, radarLayout, {{ responsive: true, displayModeBar: false }});

        const scatterTraces = {json.dumps(scatter_traces)};
        const scatterLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15, 23, 42, 0.4)',
            font: {{ color: '#94a3b8', family: 'sans-serif' }},
            xaxis: {{ title: 'Throughput (tok/s)', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
            yaxis: {{ title: 'Judge Score (0-10)', gridcolor: '#1e293b', zerolinecolor: '#334155' }},
            margin: {{ t: 30, b: 50, l: 50, r: 30 }}
        }};
        Plotly.newPlot('scatterChart', scatterTraces, scatterLayout, {{ responsive: true, displayModeBar: false }});

        // Click-to-sort logic for leaderboard table
        function sortTable(n) {{
            const table = document.getElementById("metricsTable");
            let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            switching = true;
            dir = "desc";
            while (switching) {{
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {{
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[n];
                    y = rows[i + 1].getElementsByTagName("TD")[n];
                    let xVal = parseFloat(x.innerText.replace(/[^0-9.-]+/g, "")) || x.innerText.toLowerCase();
                    let yVal = parseFloat(y.innerText.replace(/[^0-9.-]+/g, "")) || y.innerText.toLowerCase();
                    if (dir === "asc") {{
                        if (xVal > yVal) {{ shouldSwitch = true; break; }}
                    }} else if (dir === "desc") {{
                        if (xVal < yVal) {{ shouldSwitch = true; break; }}
                    }}
                }}
                if (shouldSwitch) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                }} else {{
                    if (switchcount === 0 && dir === "desc") {{
                        dir = "asc";
                        switching = true;
                    }}
                }}
            }}
        }}
    </script>
</body>
</html>"""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)

        print(f"[ReportGenerator] ✓ HTML Master Report saved → '{output_path}'")
        return output_path

    # Backwards-compatibility alias
    def generate_md(self, evals_dir: str, output_path: str = "master_report.md") -> None:
        """Fallback alias that delegates to generate_html."""
        target_html = output_path.replace(".md", ".html")
        self.generate_html(evals_dir, target_html)


# =============================================================================
# 4. SENTRY OBSERVABILITY & BENCHMARK TRACKER
# Handles Sentry Releases, Performance Tracing across video durations,
# domain-specific custom measurements, and time-series metrics.
# =============================================================================

from contextlib import contextmanager

try:
    import sentry_sdk
    _SENTRY_SDK_AVAILABLE = True
except ImportError:
    _SENTRY_SDK_AVAILABLE = False


class SentryTracker:
    """
    Manages Sentry telemetry for the OASIS benchmark.
    Powers the 3 core comparisons:
      1. Video-to-Video Duration Scaling (Performance transactions)
      2. Model-to-Model Head-to-Head Comparison (Custom measurements & metrics)
      3. Intra-Model Trade-Off Profiles (Speed vs Memory vs Accuracy)
    """

    def __init__(self) -> None:
        self.is_active: bool = False

    def init(self) -> None:
        """Initialise Sentry SDK with the configured Release tag."""
        if not _SENTRY_SDK_AVAILABLE:
            print("[SentryTracker] ℹ sentry-sdk not installed. Running in offline mode.")
            return

        from config import SENTRY_CONFIG
        if not SENTRY_CONFIG.is_enabled:
            print("[SentryTracker] ℹ Sentry DSN not configured. Running in offline mode.")
            return

        try:
            sentry_sdk.init(
                dsn=SENTRY_CONFIG.DSN,
                release=SENTRY_CONFIG.RELEASE,
                environment=SENTRY_CONFIG.ENVIRONMENT,
                traces_sample_rate=SENTRY_CONFIG.TRACES_SAMPLE_RATE,
            )
            self.is_active = True
            print(f"[SentryTracker] ✓ Sentry initialized successfully (Release: {SENTRY_CONFIG.RELEASE})")
        except Exception as exc:
            print(f"[SentryTracker] ⚠ Failed to initialize Sentry: {exc}")
            self.is_active = False

    @contextmanager
    def track_meeting(self, meeting_id: str, audio_duration_sec: float = 0.0, slide_count: int = 0):
        """
        Profiles a meeting run and categorizes it into duration tiers.
        Powers Comparison 1 (Video to Video scaling).
        """
        if not self.is_active:
            yield None
            return

        # Calibrated meeting duration tiers for the AMI corpus (20-40m range)
        if audio_duration_sec < 1200:
            duration_tier = "Short (<20m)"
        elif audio_duration_sec < 1800:
            duration_tier = "Medium (20-30m)"
        elif audio_duration_sec <= 2400:
            duration_tier = "Large (30-40m)"
        else:
            duration_tier = "Extra Large (>40m)"

        with sentry_sdk.start_transaction(op="meeting_benchmark", name=f"Meeting_A{meeting_id}") as tx:
            tx.set_tag("meeting_id", f"A{meeting_id}")
            tx.set_tag("duration_tier", duration_tier)
            tx.set_context("dataset_properties", {
                "audio_duration_sec": round(audio_duration_sec, 1),
                "slide_count": slide_count,
            })
            yield tx

    @contextmanager
    def track_stage(self, stage_op: str, description: str):
        """Profiles the execution duration of a specific pipeline stage."""
        if not self.is_active:
            yield None
            return

        with sentry_sdk.start_span(op=stage_op, description=description) as span:
            yield span

    def capture_exception(self, exc: Exception) -> None:
        """Safely report unexpected errors to Sentry with full traceback."""
        if self.is_active:
            sentry_sdk.capture_exception(exc)


# Singleton instance ready to use across the project
SENTRY_TRACKER: SentryTracker = SentryTracker()