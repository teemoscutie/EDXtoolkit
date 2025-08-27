# -*- coding: utf-8 -*-
"""Interactive EDX spectrum explorer.

This single-file **Flask + Dash** application makes it easy to load, process
and annotate energy-dispersive X-ray spectra (EDX). The GUI lets you

* drag-and-drop vendor-specific CSV files,
* smooth and background-correct the spectrum,
* detect peaks *or* fit them with a pseudo-Voigt profile,
* match peaks against reference line-energy libraries (NIST + a small manual
  supplement),
* assign and remove candidate lines interactively, and
* export the candidate table as CSV.

Typical usage
-------------
Run the script and open the internal url printed in the terminal in your browser:

    $ python edx_app.py

Coding style
------------
* PEP 8 compliant (≤ 79 characters per line where practical).
* Docstrings follow the **Google style**
* All public callables are documented.
"""

from __future__ import annotations

# ---- standard library -----------------------------------------------------
import csv
import io
import logging
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Tuple

# ---- third-party ----------------------------------------------------------
import dash  # type: ignore
import numpy as np
import pandas as pd
import socket
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, callback_context
from flask import Flask, redirect  # type: ignore
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

# Optional: PyXRay – provides line energies + notations
try:
    import pyxray  # type: ignore
    HAVE_PYXRAY = True
except Exception:
    HAVE_PYXRAY = False

print("Dash version:", dash.__version__)

# ---------------------------------------------------------------------------
# Configuration and constants
# ---------------------------------------------------------------------------

# Configure root logging for the app. INFO is a sensible default in a GUI app.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("edx_app")

# Optional default files that are attempted on startup (safe fallback exists)
CSV_PATH = '/user/name/your/path/to/csv/sample.csv'
REF_LIB   = '/user/name/your/path/to/csv/nist_all_energies.csv'

# Gaussian sigma (in **index units**, i.e., samples) used for coarse background
DEFAULT_BG_SIGMA = 30.0

# Qualitative color cycle for plotted candidate markers
CANDIDATE_COLORS = px.colors.qualitative.Plotly

#: Global, GUI-controlled parameters (persisted in memory during the session)
PARAMS: MutableMapping[str, float | int | bool] = {
    "smoothing_window": 10,          # Savitzky–Golay window length (odd)
    "savgol_polyorder": 3,           # Savitzky–Golay polynomial order
    "smooth_zero_dropouts": True,    # interpolate single zero-count outliers
    "peak_height": 10,               # find_peaks: minimum height (in counts)
    "peak_prominence": 50,           # find_peaks: minimum prominence
    "pvoigt_fraction": 0.5,          # pseudo-Voigt mixing factor (eta)
    "match_tolerance": 70,           # line matching tolerance (eV)
    "use_fitting": True,             # prefer local pVoigt fits over detection
    "background_scale": 1.0,         # scale of background to subtract
    "subtract_background": False,    # enable background subtraction
}

# ---------------------------------------------------------------------------
# Helpers: loading spectra and references
# ---------------------------------------------------------------------------

def load_pyxray_reference(
    transitions: List[str] | None = None,
    zmin: int = 3,
    zmax: int = 98,
) -> Dict[str, float]:
    """Build a *label → energy(keV)* mapping using **PyXRay**.

    PyXRay (if available) serves as an additional line-energy source. We
    generate concise Siegbahn-style labels (e.g. "Si Kα1").

    Args:
        transitions: Optional list of Siegbahn transitions to include. If
            omitted, a pragmatic default set is used (common, robust lines).
        zmin: Minimum atomic number (inclusive).
        zmax: Maximum atomic number (inclusive).

    Returns:
        A dict like ``{"Si Kα1": 1.740, ...}`` with energies in keV.
        Returns an empty dict if PyXRay is not available.
    """
    if not HAVE_PYXRAY:
        return {}

    if transitions is None:
        # Reasonable default: frequently used, robust transitions across shells
        transitions = ["Ka1", "Ka2", "Kb1", "La1", "Lb1", "Lb2", "Lg1", "Ma1", "Mb1"]

    lib: Dict[str, float] = {}
    for Z in range(zmin, zmax + 1):
        for t in transitions:
            try:
                # XrayLine object exposes Siegbahn label and energy
                line = pyxray.xray_line(Z, t)
                label = line.siegbahn          # e.g. "Si Kα1"
                energy_keV = float(line.energy_eV) / 1000.0
                # Some databases include duplicates (different references). We
                # overwrite deterministically; that is fine for this use case.
                lib[label] = energy_keV
            except Exception:
                # Transition may not exist for this element; simply skip it.
                continue
    return lib


def load_spectrum_csv(path_or_buffer: Path | io.TextIOBase) -> pd.DataFrame:
    """Parse a vendor CSV into a two-column DataFrame (``Energy``, ``Counts``).

    The parser is designed to be **robust** against varied vendor formats:
    it locates the first numeric data row, supports comma and semicolon
    separators, and accepts either a filesystem path or an already opened
    text/bytes buffer (e.g. from Dash uploads).

    Args:
        path_or_buffer: A :class:`pathlib.Path` or an open text/bytes buffer.

    Raises:
        FileNotFoundError: If a path was given and the file does not exist.
        ValueError: If no numeric data row could be located.

    Returns:
        A tidy :class:`pandas.DataFrame` with columns ``Energy`` [keV] and
        ``Counts`` (numeric, NaNs dropped).
    """
    if isinstance(path_or_buffer, Path):
        if not path_or_buffer.exists():
            raise FileNotFoundError(f"Spectrum file not found: {path_or_buffer}")
        with path_or_buffer.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    else:
        # Uploaded buffer (text or bytes)
        if hasattr(path_or_buffer, "readlines"):
            lines = path_or_buffer.readlines()
        else:  # Dash StringIO wrapper provides .getvalue()
            lines = path_or_buffer.getvalue().splitlines()
        # Ensure str lines (decode bytes if necessary)
        lines = [l if isinstance(l, str) else l.decode("utf-8") for l in lines]

    # ---- locate header / separator ---------------------------------------
    header_idx: int | None = None
    sep: str | None = None
    for i, line in enumerate(lines):
        if "Energy" in line:
            header_idx = i
            sep = ";" if ";" in line else ","
            break

    # ---- read numeric block ----------------------------------------------
    if header_idx is not None:
        # Header with column names was found; read from there
        df = pd.read_csv(io.StringIO("".join(lines)), sep=sep, skiprows=header_idx)
        df = df.iloc[:, :2]  # first two columns only
        df.columns = ["Energy", "Counts"]
    else:
        # Try delimiter auto-detection on a small sample
        try:
            sniffed = csv.Sniffer().sniff("".join(lines[:10]))
            sep = sniffed.delimiter
        except csv.Error:
            sep = ","
        data_idx: int | None = None
        for i, line in enumerate(lines):
            if len(line.split(sep)) < 2:
                continue
            try:
                float(line.split(sep)[0])
                float(line.split(sep)[1])
                data_idx = i
                break
            except ValueError:
                continue
        if data_idx is None:
            raise ValueError("No numeric data row found in CSV upload.")
        df = pd.read_csv(
            io.StringIO("".join(lines)), sep=sep, skiprows=data_idx, header=None,
            names=["Energy", "Counts"],
        )

    # convert units and drop NaNs
    df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce") / 1000.0  # eV → keV
    df["Counts"] = pd.to_numeric(df["Counts"], errors="coerce")
    df = df.dropna(subset=["Energy", "Counts"]).reset_index(drop=True)
    return df[["Energy", "Counts"]]


def load_reference(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse the NIST line-energy CSV and merge it with a manual supplement.

    The NIST export typically contains a five-row header which is skipped.
    Column names are normalized if their number matches the expected layout.
    A small manual list of light-element *Kα* and heavy-metal *Mα* lines is
    appended to cover very common cases.

    Args:
        path: Absolute path to the NIST CSV.

    Raises:
        FileNotFoundError: If *path* cannot be found.

    Returns:
        ``{"NIST": {...}, "Manual": {...}}`` – two nested mappings label → keV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Reference library not found: {path}")

    # Try delimiter auto-detection
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        delim = csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        LOGGER.warning("Delimiter sniffing failed for %s; defaulting to comma.", path)
        delim = ","

    df = pd.read_csv(path, delimiter=delim, skiprows=5)
    expected = [
        "Element", "A", "Transition", "Theory_eV", "Unc_eV",
        "Direct_eV", "Unc_Direct_eV", "Combined_eV", "Unc_Combined_eV",
        "Vapor_eV", "Unc_Vapor_eV", "Blend", "Ref",
    ]
    if len(df.columns) >= len(expected):
        df.columns = expected + list(df.columns[len(expected):])

    df = df[df.Theory_eV.notna() & df.Transition.notna() & df.Element.notna()].copy()
    df["Energy_keV"] = df["Theory_eV"] / 1000.0

    nist = {
        f"{row.Element} {row.Transition}": float(row.Energy_keV)
        for _, row in df.iterrows()
    }
    manual = {
        "C Ka": 0.277, "N Ka": 0.392, "O Ka": 0.525,
        "Os Mα1": 1.910, "Ir Mα1": 1.980, "Pt Mα1": 2.051, "Au Mα1": 2.123,
    }
    return {"NIST": nist, "Manual": manual}


# ---------------------------------------------------------------------------
# Pre-processing, peak detection / fitting and matching
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Return a processed copy of *df* according to ``PARAMS``.

    Steps
    -----
    1. Savitzky–Golay smoothing (if window and order are valid).
    2. Replace isolated zero-count points surrounded by non-zeros.
    3. Estimate and subtract a Gaussian background (optional).

    Args:
        df: Raw input spectrum with columns ``Energy`` and ``Counts``.

    Returns:
        Processed copy of the input. ``Counts`` are cast to float64.
    """
    out = df.copy()
    # Ensure float dtype to avoid assignment issues later on
    out["Counts"] = pd.to_numeric(out["Counts"], errors="coerce").astype("float64")

    w = int(PARAMS["smoothing_window"])
    p = int(PARAMS["savgol_polyorder"])
    if w > p and w % 2 == 1 and w >= 3:
        out["Counts"] = savgol_filter(out["Counts"].values, w, p, mode="interp")

    if bool(PARAMS["smooth_zero_dropouts"]):
        counts = out["Counts"]
        mask = (counts == 0) & (counts.shift(1) != 0) & (counts.shift(-1) != 0)
        out.loc[mask, "Counts"] = (counts.shift(1) + counts.shift(-1)) / 2.0

    if bool(PARAMS["subtract_background"]):
        sigma = float(DEFAULT_BG_SIGMA)
        bg = gaussian_filter1d(out["Counts"].values, sigma=sigma)
        out["Counts"] = out["Counts"] - float(PARAMS["background_scale"]) * bg
        out["Counts"] = np.clip(out["Counts"], 0, None)

    return out


def detect_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Detect peaks using :func:`scipy.signal.find_peaks`.

    Args:
        df: Pre-processed spectrum.

    Returns:
        DataFrame with columns ``center`` [keV], ``amplitude`` [counts] and
        ``width`` (an ad-hoc width estimate derived from the prominence)
    """
    x = df["Energy"].to_numpy()
    y = df["Counts"].to_numpy()
    idx, _ = find_peaks(
        y,
        height=float(PARAMS["peak_height"]),
        prominence=float(PARAMS["peak_prominence"]),
    )
    centers = x[idx]
    amps = y[idx]
    widths = [float(PARAMS["peak_prominence"]) ] * len(idx)
    return pd.DataFrame({"center": centers, "amplitude": amps, "width": widths})


def pseudo_voigt(
    x: np.ndarray,
    amplitude: float,
    center: float,
    sigma: float,
    eta: float,
) -> np.ndarray:
    """Evaluate an *unnormalized* pseudo-Voigt profile.

    A pseudo-Voigt is a linear combination of a Gaussian and a Lorentzian of
    identical FWHM.

    Args:
        x: 1-D energy axis [keV].
        amplitude: Peak height.
        center: Peak position [keV].
        sigma: Shared *σ* parameter.
        eta: Mixing fraction (0 = pure Gaussian, 1 = pure Lorentzian).

    Returns:
        Intensity values corresponding to *x*.
    """
    gaussian = amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
    lorentzian = amplitude / (1.0 + ((x - center) / sigma) ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def fit_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Fit all detected peaks with a local pseudo-Voigt profile.

    The center positions obtained from :func:`detect_peaks` serve as initial
    guesses. Fits are restricted to ±50 eV around each guess.

    Args:
        df: Pre-processed spectrum.

    Returns:
        One row per fitted peak containing the best-fit parameters, their
        1σ uncertainties (from the covariance matrix) and the coefficient of
        determination *R²*.
    """
    x = df["Energy"].to_numpy()
    y = df["Counts"].to_numpy()

    idx, _ = find_peaks(
        y,
        height=float(PARAMS["peak_height"]),
        prominence=float(PARAMS["peak_prominence"]),
    )

    results: List[Dict[str, float]] = []
    for i in idx:
        c_guess = float(x[i])
        a_guess = float(y[i])
        sigma_guess = 0.01
        eta_guess = float(PARAMS["pvoigt_fraction"])

        # Fit window ±0.05 keV around the initial guess
        mask = (x >= c_guess - 0.05) & (x <= c_guess + 0.05)
        if int(mask.sum()) < 5:
            continue

        try:
            popt, pcov = curve_fit(
                pseudo_voigt,
                x[mask],
                y[mask],
                p0=[a_guess, c_guess, sigma_guess, eta_guess],
                bounds=(
                    [0.0, c_guess - 0.02, 0.001, 0.0],
                    [np.inf, c_guess + 0.02, 0.1, 1.0],
                ),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))

            # Coefficient of determination on the fit window
            fit_y = pseudo_voigt(x[mask], *popt)
            residuals = y[mask] - fit_y
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((y[mask] - np.mean(y[mask]))**2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            results.append(
                {
                    "center": float(popt[1]),
                    "center_err": float(perr[1]),
                    "amplitude": float(popt[0]),
                    "amplitude_err": float(perr[0]),
                    "width": float(popt[2]),
                    "width_err": float(perr[2]),
                    "eta": float(popt[3]),
                    "eta_err": float(perr[3]),
                    "r_squared": r2,
                }
            )
        except (RuntimeError, ValueError) as exc:
            LOGGER.debug("Peak fit failed at x≈%.4f keV: %s", c_guess, exc)
            continue

    return pd.DataFrame(results)


def match_peaks(
    peaks: pd.DataFrame,
    libs: Mapping[str, Mapping[str, float]],
    tolerance_eV: float,
) -> pd.DataFrame:
    """Match peak centers against multiple line-energy libraries.

    Args:
        peaks: Output of :func:`detect_peaks` **or** :func:`fit_peaks` (must
            contain a ``center`` column in keV).
        libs: Nested mapping ``{"LibName": {label: energy_keV, …}, …}``.
        tolerance_eV: Maximum allowed deviation between peak and library line
            in **electron-volt**.

    Returns:
        DataFrame with columns ``center`` and ``candidates`` where
        *candidates* is a list of ``(label, energy_keV, delta_keV)`` tuples.
    """
    tol_keV = float(tolerance_eV) / 1000.0
    rows: List[Dict[str, object]] = []
    for _, r in peaks.iterrows():
        mu = float(r["center"])
        candidates: List[Tuple[str, float, float]] = []
        for lib in libs.values():
            for label, energy in lib.items():
                delta = abs(float(energy) - mu)
                if delta <= tol_keV:
                    candidates.append((label, float(energy), float(delta)))
        rows.append({"center": mu, "candidates": candidates})
    return pd.DataFrame(rows)


def get_all_element_lines(
    element: str, libs: Mapping[str, Mapping[str, float]]
) -> List[Tuple[str, float]]:
    """Return every reference line of *element* contained in *libs*.

    The function scans all provided libraries and collects any label that
    begins with the given element symbol followed by a space.
    """
    lines: List[Tuple[str, float]] = []
    for lib in libs.values():
        for label, energy in lib.items():
            if label.startswith(element + " "):
                lines.append((label, float(energy)))
    return lines


def pileup_candidates_from_peaks(
    peaks: pd.DataFrame, tol_eV: float = 70.0
) -> List[Tuple[float, Tuple[float, float]]]:
    """Coarse pile-up search: test sums E_i + E_j against existing centers.

    Args:
        peaks: Peak table with a ``center`` column in keV.
        tol_eV: Absolute tolerance for matching a sum to an existing center.

    Returns:
        List of tuples ``(sum_energy_keV, (e_i, e_j))`` that approximately
        match a detected peak center within the tolerance.
    """
    if peaks is None or peaks.empty or "center" not in peaks.columns:
        return []
    tol_keV = float(tol_eV) / 1000.0
    centers = np.sort(peaks["center"].to_numpy(dtype=float))
    out: List[Tuple[float, Tuple[float, float]]] = []
    for i in range(len(centers)):
        for j in range(i, len(centers)):
            s = centers[i] + centers[j]
            if np.any(np.isclose(centers, s, atol=tol_keV)):
                out.append((float(s), (float(centers[i]), float(centers[j]))))
    return out


# ---------------------------------------------------------------------------
# Data Initialization
# ---------------------------------------------------------------------------

# 1) Attempt to load an initial spectrum (safe fallback to empty table)
try:
    RAW_DF = load_spectrum_csv(CSV_PATH)
except Exception as exc:
    LOGGER.error("Failed to load spectrum: %s", exc)
    RAW_DF = pd.DataFrame({"Energy": [], "Counts": []})

# 2) Line references: NIST + manual supplement
try:
    REFERENCE_LIBS = load_reference(REF_LIB)
except Exception as exc:
    LOGGER.error("Failed to load reference library: %s", exc)
    REFERENCE_LIBS = {"Manual": {}}

# 3) Optionally add PyXRay (if installed)
try:
    pyx_lib = load_pyxray_reference()
    if pyx_lib:
        # Insert PyXRay first so it gets priority on label collisions
        REFERENCE_LIBS = {"PyXRay": pyx_lib, **REFERENCE_LIBS}
        LOGGER.info("Loaded PyXRay reference with %d lines.", len(pyx_lib))
    else:
        LOGGER.info("PyXRay not available or returned empty library.")
except Exception as exc:
    LOGGER.warning("PyXRay reference loading failed: %s", exc)

# 4) Slider scaling helper for peak height UI (fallback to 100 if empty)
MAX_COUNTS = (max(100, int(RAW_DF["Counts"].max())) if not RAW_DF.empty else 100)


# ---------------------------------------------------------------------------
# Dash/Flask App Setup
# ---------------------------------------------------------------------------

# Minimal external style: Inter font for a clean UI
external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap"
]


def create_slider_dynamic(id, min_val, max_val, step, value):
    """Create a Slider with a small number of evenly spaced tick marks.

    The number of ticks is fixed to ~5 for clarity in compact sidebars.
    """
    range_val = max_val - min_val
    # Fixed at 5 ticks
    num_ticks = 5
    tick_values = np.linspace(min_val, max_val, num_ticks, dtype=int)
    marks = {int(v): str(v) for v in tick_values}
    return dcc.Slider(id=id, min=min_val, max=max_val, step=step, value=value, marks=marks)


# Flask application (WSGI) and Dash application (mounted under /ui/)
FLASK_APP = Flask(__name__)
APP = dash.Dash(
    __name__, server=FLASK_APP, routes_pathname_prefix="/ui/", external_stylesheets=external_stylesheets
)


# Try to inline custom CSS if present; silently ignore if missing
try:
    css = Path("style.css").read_text(encoding="utf-8")
except Exception:
    css = ""

# Extra CSS to prevent plots from vertically stretching the page
css_extra = """
/* --- assistant fixed: prevent vertical stretching --- */
.js-plotly-plot, .dash-graph { max-height: 70vh; }
.container > *:last-child { margin-bottom: 0; }
"""

# Customize the base HTML to inline CSS without separate assets handling
from string import Template

APP.index_string = Template("""
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>EDXtoolkit</title>
    {%favicon%}
    {%css%}
    <style>$css</style>
    <style>$css_extra</style>
    <style>
      /* --- page footer, semi-transparent --- */
      .page-footer {
        text-align: center;
        margin-top: 40px;
        padding: 12px 0;
        font-size: 0.85rem;
        color: rgba(0, 0, 0, 0.5);
      }
      .page-footer a {
        color: rgba(0, 0, 0, 0.5);
        text-decoration: none;
      }
      .page-footer a:hover {
        color: #BF5AF2; /* accent colour */
        text-decoration: underline;
      }
    </style>
  </head>
  <body>
    {%app_entry%}

    <!-- Footer -->
    <div class="page-footer">
      <span>Bachelor Thesis Project by Linh Zimmermann (August 2025)</span>
      <span> | </span>
      <a href="mailto:linh.zm@.com" rel="noopener noreferrer">Contact</a>
      <span> | </span>
      <a href="https://github.com/teemoscutie/EDXtoolkit" target="_blank" rel="noopener noreferrer">GitHub</a>
    </div>

    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
""").substitute(css=css, css_extra=css_extra)
# ---- Layout ---------------------------------------------------------------
APP.layout = html.Div(
    className="app-container",
    children=[
        # 1) Upload section
        html.Div(
            className="section",
            children=[
                html.Div("Upload EDX Spectrum", className="section-title"),
                dcc.Upload(
                    id="upload-spectra",
                    className="dash-uploader",
                    children=html.Div([
                        "📄 Drag & Drop or ",
                        html.A("Select File", style={"color": "#29406b", "font-weight": 600}),
                    ]),
                    multiple=False,
                    accept=".csv",
                ),
                # Status message (set by the upload callback)
                html.Div(id="upload-feedback"),
            ],
        ),

        # 2) Main flex container (controls + graph)
        html.Div(
            className="section analysis-flex",
            children=[
                # --- Controls (left) ------------------------------------------------
                html.Div(
                    className="analysis-controls",
                    children=[
                        html.Div("Analysis Settings", className="section-title"),
                        html.Label("Smoothing window:", className="control-label"),
                        create_slider_dynamic("slider-smooth", 3, 51, 2, 11),

                        html.Label("Minimum peak height:", className="control-label", style={"margin-top": "16px"}),
                        create_slider_dynamic("slider-height", 0, np.log(MAX_COUNTS), 1, 10),

                        html.Label("Peak prominence:", className="control-label", style={"margin-top": "16px"}),
                        create_slider_dynamic("slider-prom", 1, 100, 1, 50),

                        html.Label("Matching tolerance (eV):", className="control-label", style={"margin-top": "16px"}),
                        create_slider_dynamic("slider-tol", 1, 200, 1, 70),

                        html.Label("Background subtraction scale:", className="control-label", style={"margin-top": "16px"}),
                        create_slider_dynamic("slider-bg", 0, 2, 0.1, 1.0),

                        dcc.Checklist(
                            id="check-log",
                            options=[{"label": "Log Y-axis", "value": "log"}],
                            value=[],
                            style={"margin-top": "16px"},
                        ),
                        dcc.Checklist(
                            id="check-fitting",
                            options=[
                                {"label": "Fit instead of simple peak detection", "value": "fit"},
                                {"label": "Subtract background", "value": "bg"},
                            ],
                            value=["fit"],
                            style={"margin-top": "8px"},
                        ),

                        html.Button("💾 Save all Candidates", id="save-candidates-btn", style={"margin-top": "18px"}),
                        html.Div(id="save-candidates-message"),
                        dcc.Download(id="download-candidates-csv"),

                        # --- Pile-up controls -------------------------------------
                        html.Hr(),
                        html.Div("Pile-up detection", className="section-title"),
                        dcc.Checklist(
                            id="check-pileup",
                            options=[{"label": "Detect pile-ups (for log)", "value": "on"}],
                            value=[],
                        ),
                        html.Button("➕ Add detected pile-ups", id="add-pileups-btn", style={"marginTop": "8px"}),
                        dcc.Dropdown(
                            id="remove-pileup-selector",
                            placeholder="Select pile-up to remove...",
                            multi=False,
                            style={"width": "92%", "margin": "12px 0"},
                        ),
                        html.Button("❌ Remove Pile-up", id="remove-pileup-btn", n_clicks=0, style={"margin-left": "10px"}),
                    ],
                ),

                # --- Graph & interaction (right) -----------------------------------
                html.Div(
                    className="analysis-graph",
                    children=[
                        html.Div("Spectrum & Peaks", className="section-title"),
                        dcc.Graph(
                            id="spectrum-graph",
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "scrollZoom": False,
                                "modeBarButtonsToRemove": [
                                    "select2d", "lasso2d", "autoScale2d",
                                    "zoom2d", "zoomIn2d", "zoomOut2d", "resetScale2d",
                                    "hoverCompareCartesian", "hoverClosestCartesian",
                                ],
                            },
                        ),
                        dcc.Dropdown(
                            id="manual-peak-selector",
                            placeholder="Select candidate...",
                            multi=False,
                            style={"width": "92%", "margin": "12px 0"},
                        ),
                        html.Div(
                            id="selection-log",
                            style={
                                "whiteSpace": "pre-wrap",
                                "padding": "10px",
                                "fontSize": "14px",
                                "maxHeight": "30vh",
                                "overflow": "auto",
                            },
                        ),
                        dcc.Dropdown(
                            id="remove-persisted-selector",
                            placeholder="Select peak to remove...",
                            multi=False,
                            style={"width": "92%", "margin": "12px 0"},
                        ),
                        html.Button("❌ Remove Peak", id="remove-persisted-btn", n_clicks=0, style={"margin-left": "10px"}),
                    ],
                ),
            ],
        ),

        # 3) Client-side state stores
        dcc.Store(id="uploaded-spectrum-store", storage_type="memory"),
        dcc.Store(id="uploaded-spectrum-filename", storage_type="memory"),
        dcc.Store(id="candidates-store", storage_type="memory"),
        dcc.Store(id="persisted-candidates", data=[]),
        dcc.Store(id="candidate-store"),
        dcc.Store(id="persisted-pileups", data=[]),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ---- Upload Callback (store file content & name) --------------------------
@APP.callback(
    Output('uploaded-spectrum-store', 'data'),
    Output('uploaded-spectrum-filename', 'data'),
    Output('upload-feedback', 'children'),
    Input('upload-spectra', 'contents'),
    State('upload-spectra', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    """Decode the uploaded CSV and normalize it to Energy/Counts JSON.

    Returns a user-visible success/error message. Uses the robust CSV parser
    as a fallback if required columns are absent.
    """
    if not contents:
        return dash.no_update, dash.no_update, ''
    import base64, io
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8', errors='ignore')
        # Attempt 1: read directly (expecting headers)
        df = pd.read_csv(io.StringIO(decoded))
        if not all(col in df.columns for col in ["Energy", "Counts"]):
            # Fallback: robust vendor CSV parsing
            df = load_spectrum_csv(io.StringIO(decoded))
        return df.to_json(orient="split", date_format="iso"), filename, html.Span(f"✅ {filename} uploaded successfully!", className="upload-success")
    except Exception as e:
        return dash.no_update, dash.no_update, html.Span(f"❌ Error: {str(e)}", className="upload-error")


@APP.callback(
    Output("spectrum-graph", "figure"),
    Output("selection-log", "children"),
    Input("slider-smooth", "value"),
    Input("slider-height", "value"),
    Input("slider-prom", "value"),
    Input("slider-tol", "value"),
    Input("slider-bg", "value"),
    Input("check-log", "value"),
    Input("check-fitting", "value"),
    Input("persisted-candidates", "data"),
    Input("candidate-store", "data"),
    Input("manual-peak-selector", "value"),
    Input("check-pileup", "value"),
    Input("persisted-pileups", "data"),
    State("uploaded-spectrum-store", "data"),
)
def update_figure(
    smooth, height, prom, tol, bg_scale,
    log_opts, fit_opts,
    persisted, candidate_store, selected_val,
    check_pileup_vals, persisted_pileups,
    uploaded_df_json,
):
    """Render the spectrum and interaction overlays based on UI state.

    This is the central UI callback: it updates global PARAMS, preprocesses
    the current spectrum, detects or fits peaks, matches lines, and draws
    markers (peaks, assignments, optional pile-ups). It also compiles a short
    textual log for the sidebar.
    """
    df = pd.read_json(uploaded_df_json, orient='split') if uploaded_df_json else RAW_DF

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="RAW_DF is empty! (No spectrum loaded or parsing error.)",
            xref="paper", yref="paper", showarrow=False, font=dict(size=18, color="red"),
        )
        return fig, "RAW_DF is empty!"

    # ---- Update PARAMS from UI -------------------------------------------
    PARAMS["smoothing_window"] = int(smooth) if smooth is not None else 11
    PARAMS["peak_height"] = int(height) if height is not None else 10
    PARAMS["peak_prominence"] = int(prom) if prom is not None else 50
    PARAMS["match_tolerance"] = int(tol) if tol is not None else 70
    PARAMS["use_fitting"] = "fit" in (fit_opts or [])
    PARAMS["subtract_background"] = "bg" in (fit_opts or [])
    PARAMS["background_scale"] = float(bg_scale) if bg_scale is not None else 1.0

    # ---- Processing & peak table -----------------------------------------
    proc_df = preprocess(df)
    peaks_df = fit_peaks(proc_df) if PARAMS["use_fitting"] else detect_peaks(proc_df)
    matched_df = match_peaks(peaks_df, REFERENCE_LIBS, float(PARAMS["match_tolerance"]))

    # ---- Pile-up candidates (computed for log only unless persisted) -----
    pileups_detected: List[Tuple[float, Tuple[float, float]]] = []
    if "on" in (check_pileup_vals or []):
        pileups_detected = pileup_candidates_from_peaks(peaks_df, float(PARAMS["match_tolerance"]))

    # ---- Plot -------------------------------------------------------------
    x = proc_df["Energy"].to_numpy()
    y = proc_df["Counts"].to_numpy()
    fig = go.Figure()
    if "log" in (log_opts or []):
        # Avoid log(0) by clipping to 1 for display only
        y_plot = np.where(y > 0, y, 1)
        fig.add_trace(go.Scatter(x=x, y=y_plot, mode="lines", name="Processed"))
        fig.update_yaxes(type="log", title="Counts (log10)")
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Processed"))
        fig.update_yaxes(type="linear", title="Counts")

    # Detected peak markers (blue crosses at matched centers)
    for i, row in matched_df.iterrows():
        # >>> Show fitted curves when fitting is enabled <<<
        if PARAMS["use_fitting"] and not peaks_df.empty and {"amplitude","center","width","eta"}.issubset(peaks_df.columns):
            for _, r in peaks_df.iterrows():
                x_fit = np.linspace(r["center"] - 0.05, r["center"] + 0.05, 200)
                y_fit = pseudo_voigt(x_fit, r["amplitude"], r["center"], r["width"], r["eta"])
                # Auf Log-Achse nichts ändern: y_fit stammt aus dem Fit auf proc_df
                fig.add_trace(
                    go.Scatter(
                        x=x_fit, y=y_fit,
                        mode="lines",
                        line=dict(dash="dot"),
                        name=f"Fit @ {r['center']:.3f} keV",
                        showlegend=False,
                    )
                )
        peak_center = float(row["center"])
        idx_nearest = (proc_df["Energy"] - peak_center).abs().idxmin()
        y_val = float(proc_df.iloc[idx_nearest]["Counts"])
        fig.add_trace(
            go.Scatter(
                x=[peak_center], y=[y_val], mode="markers",
                marker=dict(color="blue", size=11, symbol="x"),
                name=f"Peak {i + 1}", showlegend=False,
            )
        )

    legend_labels = set()
    msg_lines: List[str] = []

    # ---- Persisted pile-ups (shown as red crosses with sum labels) --------
    if persisted_pileups:
        for p in persisted_pileups:
            s = float(p["sum_energy"])
            e1 = float(p["e1"])
            e2 = float(p["e2"])
            idx_nearest = (proc_df["Energy"] - s).abs().idxmin()
            y_val = float(proc_df.iloc[idx_nearest]["Counts"])
            fig.add_trace(
                go.Scatter(
                    x=[s], y=[y_val],
                    mode="markers+text",
                    marker=dict(symbol="x", size=12, color="red"),
                    text=[f"Pile-up {e1:.3f}+{e2:.3f}"],
                    textposition="top center",
                    name="Pile-up", showlegend=False,
                )
            )
        msg_lines.append(f"{len(persisted_pileups)} pile-up markers shown.")

    # ---- Log-only listing of freshly detected pile-ups --------------------
    if "on" in (check_pileup_vals or []):
        if pileups_detected:
            msg_lines.append(f"Pile-up candidates detected: {len(pileups_detected)}")
            for k, (s, (e1, e2)) in enumerate(pileups_detected[:8], 1):
                msg_lines.append(f"  {k:>2}. {s:.3f} keV ≈ {e1:.3f} + {e2:.3f}")
        else:
            msg_lines.append("Pile-up candidates detected: 0")

    # ---- Persisted line candidates (with horizontal error bars) -----------
    if persisted:
        for idx, cand in enumerate(persisted):
            color = CANDIDATE_COLORS[idx % len(CANDIDATE_COLORS)]

            # x-position: measured center if available, otherwise library energy
            x_pos = cand.get("measured_center")
            try:
                x_pos = float(x_pos)
                if not np.isfinite(x_pos):
                    raise ValueError
            except Exception:
                x_pos = float(cand["energy"])

            # Error bar from match delta (in keV); only show if positive & finite
            errx = cand.get("delta_keV")
            try:
                errx = float(errx)
                show_err = np.isfinite(errx) and (errx > 0.0)
            except Exception:
                errx = 0.0
                show_err = False

            label = str(cand.get("label", f"{cand.get('element','?')} @ {cand.get('energy',np.nan):.3f} keV"))

            idx_nearest = (proc_df["Energy"] - x_pos).abs().idxmin()
            y_val = float(proc_df.iloc[idx_nearest]["Counts"])

            fig.add_trace(
                go.Scatter(
                    x=[x_pos], y=[y_val],
                    mode="markers+text",
                    marker_symbol="x",
                    marker_size=19,
                    marker_color=color,
                    text=[label],
                    textposition="top center",
                    name=label,
                    showlegend=(label not in legend_labels),
                    error_x=dict(type="data", array=[errx], visible=show_err),
                )
            )
            legend_labels.add(label)
        msg_lines.append(f"{len(persisted)} manually assigned peaks.")

    # ---- Show all lines of the selected element (if any) ------------------
    if selected_val and candidate_store:
        match = [c for c in candidate_store if c["value"] == selected_val]
        if match:
            element = match[0]["element"]
            # Choose a consistent color for these auxiliary markers
            aux_color = "black"
            for line_name, line_energy in get_all_element_lines(element, REFERENCE_LIBS):
                idx_nearest = (proc_df["Energy"] - line_energy).abs().idxmin()
                y_val = float(proc_df.iloc[idx_nearest]["Counts"])
                fig.add_trace(
                    go.Scatter(
                        x=[line_energy], y=[y_val], mode="markers",
                        marker_symbol="cross", marker_size=10, marker_color=aux_color,
                        name=f"{element} line: {line_name}", showlegend=False,
                    )
                )
            msg_lines.append(f"Showing all lines for {element} (selected).")

    fig.update_layout(
        title="EDX Spectrum",
        xaxis_title="Energy (keV)",
        legend_title="Assignments",
    )
    return fig, "\n".join(msg_lines)


# -- handle_click, manage_persisted_candidates, fill_remove_dropdown: same
#    pattern; all callbacks consistently read State('uploaded-spectrum-store')

@APP.callback(
    Output("candidate-store", "data"),
    Output("manual-peak-selector", "options"),
    Input("spectrum-graph", "clickData"),
    State("slider-tol", "value"),
    State("persisted-candidates", "data"),
    State("uploaded-spectrum-store", "data"),
)
def handle_click(click_data, tol, persisted, uploaded_df_json):
    """On plot click: gather candidate line matches near the clicked x-value.

    The result is stored twice: a rich internal list (for state) and a compact
    list of options for the dropdown (for rendering only).
    """
    df = pd.read_json(uploaded_df_json, orient='split') if uploaded_df_json else RAW_DF
    if not click_data or df.empty:
        return [], []

    xval = float(click_data["points"][0]["x"])
    proc_df = preprocess(df)
    peaks_df = fit_peaks(proc_df) if PARAMS["use_fitting"] else detect_peaks(proc_df)
    matched_df = match_peaks(peaks_df, REFERENCE_LIBS, float(tol))

    dropdown_candidates = []
    already_values = set(c.get("value") for c in (persisted or []) if "value" in c)

    for peak_idx, row in matched_df.iterrows():
        center = float(row["center"])
        if abs(center - xval) >= (float(tol) / 1000.0):
            continue

        # Fit-derived measurement uncertainty (if available)
        peak_center_err = None
        measured_center = center
        if "center_err" in peaks_df.columns:
            try:
                v = float(peaks_df.iloc[int(peak_idx)].get("center_err", np.nan))
                if np.isfinite(v) and v > 0:
                    peak_center_err = v
            except Exception:
                pass
        try:
            measured_center = float(peaks_df.iloc[int(peak_idx)]["center"])
        except Exception:
            pass

        # Build dropdown entries for each candidate label near this peak
        for cand_idx, (label, energy, delta) in enumerate(row["candidates"]):
            value_id = f"{peak_idx}_{cand_idx}"
            if value_id in already_values:
                continue  # do not offer already persisted items

            element = label.split(" ")[0] if " " in label else label
            dropdown_candidates.append({
                "label": f"{label} @ {energy:.3f} keV (Δ={delta*1000:.1f} eV)",
                "value": value_id,
                "energy": float(energy),
                "element": element,
                "peak_idx": int(peak_idx),
                "cand_idx": int(cand_idx),
                "label_full": label,
                "center_err": peak_center_err,      # keV
                "measured_center": measured_center, # keV
                "delta": float(delta),
            })

    options = [{"label": o["label"], "value": o["value"]} for o in dropdown_candidates]
    return dropdown_candidates, options


@APP.callback(
    Output("persisted-pileups", "data", allow_duplicate=True),
    Output("remove-pileup-selector", "value"),
    Input("remove-pileup-btn", "n_clicks"),
    State("remove-pileup-selector", "value"),
    State("persisted-pileups", "data"),
    prevent_initial_call=True,
)
def remove_pileup(n_clicks, idx, persisted):
    """Remove a selected persisted pile-up marker by list index."""
    if not n_clicks or idx is None or not persisted:
        return dash.no_update, dash.no_update
    new_persisted = [p for i, p in enumerate(persisted) if i != idx]
    return new_persisted, None


@APP.callback(
    Output("remove-pileup-selector", "options"),
    Input("persisted-pileups", "data"),
)
def fill_remove_pileup_dropdown(persisted):
    """Provide dropdown options reflecting current persisted pile-ups."""
    if not persisted:
        return []
    return [
        {"label": p.get("label", f"Pile-up @ {p.get('sum_energy', float('nan')):.3f} keV"), "value": i}
        for i, p in enumerate(persisted)
    ]


@APP.callback(
    Output("remove-persisted-selector", "options"),
    Input("persisted-candidates", "data"),
)
def fill_remove_dropdown(persisted):
    """Provide dropdown options reflecting current persisted line candidates."""
    if not persisted:
        return []
    return [
        {"label": c.get("label", f"Peak {i+1}"), "value": i}
        for i, c in enumerate(persisted)
    ]

@APP.callback(
    Output("persisted-pileups", "data"),
    Input("add-pileups-btn", "n_clicks"),
    State("persisted-pileups", "data"),
    State("slider-tol", "value"),
    State("check-fitting", "value"),
    State("uploaded-spectrum-store", "data"),
    prevent_initial_call=True,
)
def add_pileups(n_clicks, persisted, tol, fit_opts, uploaded_df_json):
    if not n_clicks:
        return dash.no_update

    # Retrieve current spectrum
    df = pd.read_json(uploaded_df_json, orient='split') if uploaded_df_json else RAW_DF
    if df.empty:
        return persisted or []

    # Preprocess & detect peaks
    proc_df = preprocess(df)
    peaks_df = fit_peaks(proc_df) if ("fit" in (fit_opts or [])) else detect_peaks(proc_df)

    # Pile-up candidates in keV
    detected = pileup_candidates_from_peaks(peaks_df, float(tol))

    # Existing entries + de-duplication
    persisted = persisted or []
    existing_keys = {round(float(p.get("sum_energy", -1.0)), 6) for p in persisted}

    for s, (e1, e2) in detected:
        key = round(float(s), 6)
        if key in existing_keys:
            continue
        persisted.append({
            "sum_energy": float(s),
            "e1": float(e1),
            "e2": float(e2),
            "label": f"Pile-up {e1:.3f}+{e2:.3f} @ {s:.3f} keV",
        })
        existing_keys.add(key)

    return persisted

# ---- Download of candidates with dynamic filename ------------------------
@APP.callback(
    Output("download-candidates-csv", "data"),
    Input("save-candidates-btn", "n_clicks"),
    State("persisted-candidates", "data"),
    State("uploaded-spectrum-filename", "data"),  # used to build output name
    prevent_initial_call=True,
)
def save_candidates(n_clicks, candidates, uploaded_filename):
    """Serialize persisted candidates to CSV and trigger a download."""
    if not candidates:
        return dash.no_update
    import io
    df = pd.DataFrame(candidates)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    base = (uploaded_filename or "candidates").rsplit('.', 1)[0]
    outname = f"{base}_candidates.csv"
    return dict(content=csv_buf.getvalue(), filename=outname)


@APP.callback(
    Output("persisted-candidates", "data"),
    Output("remove-persisted-selector", "value"),
    Input("manual-peak-selector", "value"),
    Input("remove-persisted-btn", "n_clicks"),
    State("candidate-store", "data"),
    State("persisted-candidates", "data"),
    State("remove-persisted-selector", "value"),
    prevent_initial_call=True,
)
def manage_persisted_candidates(selected_idx, n_clicks, candidate_store, persisted, remove_idx):
    """Toggle or remove persisted manual assignments based on UI actions."""
    trig = getattr(ctx, "triggered_id", None)
    persisted = (persisted or []).copy()

    if trig == "manual-peak-selector":
        if selected_idx is None or not candidate_store:
            return dash.no_update, dash.no_update

        match = next((c for c in candidate_store if c["value"] == selected_idx), None)
        if not match:
            return dash.no_update, dash.no_update

        # Toggle by unique value id
        existing_i = next((i for i, c in enumerate(persisted) if c.get("value") == selected_idx), None)
        if existing_i is not None:
            del persisted[existing_i]
            return persisted, dash.no_update

        label = match.get("label_full") or f"{match['element']} @ {match['energy']:.3f} keV"

        # Sanitize uncertainty/position
        center_err = match.get("center_err")
        try:
            center_err = float(center_err) if center_err is not None else None
            if not np.isfinite(center_err) or center_err <= 0:
                center_err = None
        except Exception:
            center_err = None

        measured_center = match.get("measured_center")
        try:
            measured_center = float(measured_center) if measured_center is not None else None
        except Exception:
            measured_center = None

        persisted.append({
            "label": label,
            "value": match["value"],
            "energy": float(match["energy"]),
            "element": match["element"],
            "measured_center": measured_center,
            "delta_keV": float(match.get("delta", 0.0)),
        })
        return persisted, dash.no_update

    if trig == "remove-persisted-btn":
        if not n_clicks or remove_idx is None or not persisted:
            return dash.no_update, dash.no_update
        new_persisted = [c for i, c in enumerate(persisted) if i != remove_idx]
        return new_persisted, None

    return dash.no_update, dash.no_update


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


def get_local_ip():
    """Best-effort detection of the LAN IP to make the app reachable.

    This opens a UDP socket to a public IP (no data is actually sent) to let
    the OS pick the preferred outbound interface, then reads the local socket
    address. On failure, falls back to 127.0.0.1.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


local_ip = get_local_ip()


@FLASK_APP.route("/")
def index() -> str:
    """Redirect visitors of the root path to the Dash UI under ``/ui/``."""
    return redirect("/ui/")


if __name__ == "__main__":
    # Run the app on the detected local IP so other devices in the LAN may
    # connect; disable Flask debug reloader for a steadier single-process run.
    APP.run(debug=False, host=local_ip, port=5000)