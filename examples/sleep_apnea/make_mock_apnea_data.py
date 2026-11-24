#!/usr/bin/env python
"""Mock NCH and CHAT sleep studies, for NCHSleepDataset and CHATDataset.

What it shows: the on-disk layout that :class:`~pyhealth.datasets.NCHSleepDataset`
and :class:`~pyhealth.datasets.CHATDataset` read, filled with synthetic sleep
studies that :class:`~pyhealth.tasks.SleepApneaDetection` can run on.

Dataset: none needed. It writes stand-ins for the NSRR NCH Sleep DataBank and
CHAT, which are restricted (approved NSRR data request). Nothing is
downloaded. Under ``--out`` it writes ``raw/nchsdb`` (EDFs, ``.tsv``
annotations, ``health_data/`` CSVs) and ``raw/chat`` (EDFs, NSRR XML
annotations, subject CSVs), plus ``manifest.json`` with the events placed in
every study.

The signals are synthetic but respond to the annotated events the way real
signals do: airflow drops during apneas and hypopneas, breathing effort stops
during central apneas, SpO2 drops after each event, heart rate slows and then
rises, CO2 goes flat, and the EEG shows an arousal. A model can therefore learn
something from them, which makes them usable as a smoke test.

How to run, from the PyHealth repo root with PyHealth installed:

    python examples/sleep_apnea/make_mock_apnea_data.py
    python examples/sleep_apnea/make_mock_apnea_data.py --minutes 6 \
        --out examples/sleep_apnea/tiny

The first writes ``examples/sleep_apnea/out``, the second a smaller set. Then
point ``NCHSleepDataset`` at ``out/raw/nchsdb`` and ``CHATDataset`` at
``out/raw/chat``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

EPOCH_SEC = 30
NCH_STAGE = {
    "W": "Sleep stage W",
    "N1": "Sleep stage N1",
    "N2": "Sleep stage N2",
    "N3": "Sleep stage N3",
    "R": "Sleep stage R",
}
CHAT_STAGE = {
    "W": "Wake",
    "N1": "Stage 1 sleep",
    "N2": "Stage 2 sleep",
    "N3": "Stage 3 sleep",
    "R": "REM sleep",
}
NCH_EVENT = {
    "obstructive": "Obstructive Apnea",
    "central": "Central Apnea",
    "mixed": "Mixed Apnea",
    "hypopnea": "Hypopnea",
    "obs_hypopnea": "Obstructive Hypopnea",
}
CHAT_EVENT = {
    "obstructive": "Obstructive apnea",
    "central": "Central apnea",
    "mixed": "Obstructive apnea",
    "hypopnea": "Hypopnea",
    "obs_hypopnea": "Hypopnea",
}


# =============================================================== study specs
@dataclass
class StudySpec:
    dataset: str  # "nch" or "chat"
    name: str  # file stem
    patient_id: str
    ahi: float  # target respiratory events per hour of sleep
    fs: int  # EDF sampling rate
    drop_channels: tuple = ()  # simulate a study missing a required channel
    age_days: int = 0
    male: int = 0
    note: str = ""
    visit: str = "baseline"  # CHAT only: "baseline" or "followup"


def nch_specs() -> list[StudySpec]:
    # 8 patients / 10 studies. Two patients have repeat studies, one study has
    # no events (left out by SleepApneaDetection(min_ahi=3)), one lacks Capno
    # (skipped by the task unless fill_missing=True).
    rows = [
        ("10012", "21037", 18, 256, (), 2900, 1, ""),
        ("10012", "24410", 11, 256, (), 3650, 1, "repeat study"),
        ("10045", "21194", 30, 256, (), 1500, 0, ""),
        ("10063", "21388", 7, 400, (), 4800, 1, "400 Hz EDF"),
        ("10071", "22052", 25, 256, (), 2200, 0, ""),
        ("10071", "25871", 14, 256, (), 2600, 0, "repeat study"),
        ("10088", "22213", 9, 256, (), 5400, 1, ""),
        ("10093", "22540", 40, 256, (), 1100, 0, ""),
        ("10104", "23016", 0, 256, (), 3300, 1, "no events (AHI 0)"),
        ("10117", "23377", 20, 256, ("Capno",), 4100, 0, "no Capno channel"),
    ]
    return [
        StudySpec("nch", f"{p}_{s}", p, a, fs, d, age, m, n)
        for p, s, a, fs, d, age, m, n in rows
    ]


def chat_specs() -> list[StudySpec]:
    rows = [
        (300001, 22, 200, (), 6, 1, ""),
        (300002, 9, 200, (), 7, 0, ""),
        (300003, 35, 256, (), 5, 1, "256 Hz EDF"),
        (300004, 15, 200, (), 8, 0, ""),
        (300005, 6, 200, (), 9, 1, ""),
        (300006, 27, 200, (), 6, 0, ""),
        (300007, 0, 200, (), 7, 1, "no events (AHI 0)"),
        (300008, 18, 200, ("Cap",), 5, 0, "no Cap channel"),
    ]
    specs = [
        StudySpec("chat", f"chat-baseline-{i}", str(i), a, fs, d, age * 365, m, n)
        for i, a, fs, d, age, m, n in rows
    ]
    # Follow-up visits (about 7 months later) for two subjects, with XML
    # annotations only.
    specs += [
        StudySpec(
            "chat",
            "chat-followup-300001",
            "300001",
            12,
            200,
            (),
            6 * 365 + 210,
            1,
            "follow-up visit, XML annotations only",
            visit="followup",
        ),
        StudySpec(
            "chat",
            "chat-followup-300002",
            "300002",
            4,
            200,
            (),
            7 * 365 + 210,
            0,
            "follow-up visit, XML annotations only",
            visit="followup",
        ),
    ]
    return specs


# =============================================================== EDF writer
def _ascii(value, width: int) -> bytes:
    s = str(value)
    if len(s) > width:
        raise ValueError(f"{s!r} does not fit in {width} chars")
    return s.ljust(width).encode("ascii")


def _num8(x: float) -> str:
    for prec in range(8, 0, -1):
        s = f"{x:.{prec}g}"
        if len(s) <= 8:
            return s
    raise ValueError(x)


def write_edf(path: Path, channels: list[dict], start: datetime) -> None:
    """Write a 16-bit EDF file. Each channel: label, unit, fs, data (physical)."""
    n_rec = min(len(c["data"]) // c["fs"] for c in channels)
    ns = len(channels)
    digital, pmin, pmax = [], [], []
    for c in channels:
        d = np.asarray(c["data"][: n_rec * c["fs"]], dtype=float)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-6:
            lo, hi = lo - 1, hi + 1
        pad = 0.02 * (hi - lo)
        lo_s, hi_s = _num8(lo - pad), _num8(hi + pad)
        lo, hi = float(lo_s), float(hi_s)
        dig = np.round((d - lo) / (hi - lo) * 65535 - 32768).clip(-32768, 32767)
        digital.append(dig.astype("<i2").reshape(n_rec, c["fs"]))
        pmin.append(lo_s)
        pmax.append(hi_s)

    header = b"".join(
        [
            _ascii("0", 8),
            _ascii("X X X X", 80),
            _ascii(f"Startdate {start.strftime('%d-%b-%Y').upper()} X X X", 80),
            _ascii(start.strftime("%d.%m.%y"), 8),
            _ascii(start.strftime("%H.%M.%S"), 8),
            _ascii(256 * (ns + 1), 8),
            _ascii("", 44),
            _ascii(n_rec, 8),
            _ascii(1, 8),
            _ascii(ns, 4),
        ]
    )
    header += b"".join(_ascii(c["label"], 16) for c in channels)
    header += b"".join(_ascii("", 80) for _ in channels)
    header += b"".join(_ascii(c["unit"], 8) for c in channels)
    header += b"".join(_ascii(v, 8) for v in pmin)
    header += b"".join(_ascii(v, 8) for v in pmax)
    header += b"".join(_ascii(-32768, 8) for _ in channels)
    header += b"".join(_ascii(32767, 8) for _ in channels)
    header += b"".join(_ascii("", 80) for _ in channels)
    header += b"".join(_ascii(c["fs"], 8) for c in channels)
    header += b"".join(_ascii("", 32) for _ in channels)
    assert len(header) == 256 * (ns + 1)
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.concatenate(digital, axis=1).tobytes())


# =============================================================== physiology
@dataclass
class Event:
    onset: float
    duration: float
    kind: str  # obstructive | central | mixed | hypopnea | obs_hypopnea

    @property
    def end(self) -> float:
        return self.onset + self.duration

    @property
    def is_apnea(self) -> bool:
        return self.kind in ("obstructive", "central", "mixed")


@dataclass
class Study:
    spec: StudySpec
    stages: list[str]
    events: list[Event] = field(default_factory=list)

    @property
    def sleep_hours(self) -> float:
        return sum(s != "W" for s in self.stages) * EPOCH_SEC / 3600

    @property
    def ahi(self) -> float:
        return len(self.events) / self.sleep_hours if self.sleep_hours else 0.0


def make_hypnogram(n_epochs: int) -> list[str]:
    cycle = ["N1"] * 2 + ["N2"] * 6 + ["N3"] * 6 + ["N2"] * 3 + ["R"] * 5 + ["W"]
    stages = ["W"] * 3
    while len(stages) < n_epochs:
        stages += cycle
    stages = stages[:n_epochs]
    stages[-2:] = ["W", "W"]
    return stages


def place_events(study: Study, rng: np.random.Generator) -> None:
    if study.spec.ahi <= 0:
        return
    n_target = max(1, round(study.spec.ahi * study.sleep_hours))
    sleep = np.array([s != "W" for s in study.stages])
    total = len(study.stages) * EPOCH_SEC
    kinds = ["obstructive", "central", "mixed", "hypopnea", "obs_hypopnea"]
    probs = [0.30, 0.15, 0.10, 0.25, 0.20]
    for _ in range(n_target * 50):
        if len(study.events) >= n_target:
            break
        kind = rng.choice(kinds, p=probs)
        dur = round(
            float(
                rng.uniform(8, 22)
                if kind in ("obstructive", "central", "mixed")
                else rng.uniform(10, 30)
            ),
            1,
        )
        onset = round(float(rng.uniform(0, total - dur - 20)), 1)
        # event plus 10 s recovery must sit entirely in sleep
        e0, e1 = int(onset // EPOCH_SEC), int((onset + dur + 10) // EPOCH_SEC)
        if e1 >= len(sleep) or not sleep[e0 : e1 + 1].all():
            continue
        if any(
            onset < ev.end + 15 and ev.onset < onset + dur + 15 for ev in study.events
        ):
            continue
        study.events.append(Event(onset, dur, str(kind)))
    study.events.sort(key=lambda ev: ev.onset)


def _box(t, a, b, ramp=1.5):
    return np.minimum(np.clip((t - a) / ramp, 0, 1), np.clip((b - t) / ramp, 0, 1))


def _slow(n, fs, rng, step_s=2.0):
    """Smooth random curve, unit std, varying on a time scale of step_s."""
    knots = rng.standard_normal(int(n / fs / step_s) + 3)
    return np.interp(np.arange(n) / fs / step_s, np.arange(len(knots)), knots)


def _colored(n, rng, amp, pole=0.95):
    x = sps.lfilter([1.0], [1.0, -pole], rng.standard_normal(n))
    return amp * x / (x.std() + 1e-12)


def _osc(t, freq, rng, n, fs, jitter=3.0):
    return np.sin(
        2 * np.pi * freq * t + jitter * _slow(n, fs, rng, 4.0) + rng.uniform(0, 6.28)
    )


def synthesize(study: Study, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return physiological source signals (natural units) at the study's fs."""
    fs = study.spec.fs
    n = len(study.stages) * EPOCH_SEC * fs
    t = np.arange(n) / fs
    stage = np.array(study.stages)[
        np.minimum((t // EPOCH_SEC).astype(int), len(study.stages) - 1)
    ]

    def by_stage(table):
        return np.vectorize(table.get)(stage).astype(float)

    # ---- event envelopes
    flow_env, eff_env = np.ones(n), np.ones(n)
    paradox, in_event, post = np.zeros(n), np.zeros(n), np.zeros(n)
    spo2_drop = np.zeros(n)
    for ev in study.events:
        m = _box(t, ev.onset, ev.end)
        flow_env = np.minimum(flow_env, 1 - (0.95 if ev.is_apnea else 0.6) * m)
        in_event = np.maximum(in_event, m)
        half = ev.onset + ev.duration / 2
        if ev.kind == "central":
            eff_env = np.minimum(eff_env, 1 - 0.95 * m)
        elif ev.kind == "mixed":
            eff_env = np.minimum(eff_env, 1 - 0.95 * _box(t, ev.onset, half))
            paradox = np.maximum(paradox, _box(t, half, ev.end))
        elif ev.kind == "obstructive":
            paradox = np.maximum(paradox, m)
        else:
            eff_env = np.minimum(eff_env, 1 - 0.4 * m)
            if ev.kind == "obs_hypopnea":
                paradox = np.maximum(paradox, 0.5 * m)
        post = np.maximum(post, _box(t, ev.end, ev.end + 8, ramp=1.0))
        depth = rng.uniform(4, 9) if ev.is_apnea else rng.uniform(3, 5)
        spo2_drop += depth * np.exp(
            -0.5 * ((t - (ev.end + 7)) / max(4.0, 0.4 * ev.duration)) ** 2
        )
    flow_env = flow_env + 0.4 * post
    eff_env = eff_env + 0.3 * post

    # ---- breathing
    f_resp = by_stage({"W": 0.36, "N1": 0.34, "N2": 0.32, "N3": 0.30, "R": 0.37})
    f_resp = f_resp * (1 + 0.06 * _slow(n, fs, rng, 5.0))
    phase = 2 * np.pi * np.cumsum(f_resp) / fs + rng.uniform(0, 6.28)
    breath = np.sin(phase)
    src = {
        "ptaf": 0.6 * flow_env * np.sign(breath) * np.abs(breath) ** 0.7
        + _colored(n, rng, 0.02, 0.5),
        "airflow": 0.5 * flow_env * np.sin(phase - 0.4) + _colored(n, rng, 0.02, 0.5),
        "thoracic": 0.8 * eff_env * breath + _colored(n, rng, 0.03),
        "abdominal": 0.7 * eff_env * np.sin(phase + 0.2 + 0.8 * np.pi * paradox)
        + _colored(n, rng, 0.03),
    }
    co2_level = 38 + 4 * in_event + 1.5 * _slow(n, fs, rng, 30.0)
    co2_wave = 1 / (1 + np.exp(-10 * (np.sin(phase - 1.0) - 0.2)))
    src["capno"] = co2_level * co2_wave * np.clip(flow_env, 0, 1) + _colored(
        n, rng, 0.4, 0.5
    )
    src["spo2"] = np.clip(
        97.5
        + 0.4 * _slow(n, fs, rng, 20.0)
        - spo2_drop
        + 0.05 * rng.standard_normal(n),
        70,
        100,
    )

    # ---- ECG (uV): heart slows during an event and speeds up afterwards
    hr = by_stage({"W": 95, "N1": 90, "N2": 85, "N3": 80, "R": 92})
    hr = (
        hr * (1 - 0.08 * in_event + 0.15 * post)
        + 3 * breath
        + 2 * _slow(n, fs, rng, 10.0)
    )
    beats = np.flatnonzero(np.diff(np.floor(np.cumsum(hr / 60) / fs)) > 0)
    ecg = np.zeros(n)
    waves = [
        (150, -0.20, 0.025),
        (-100, -0.03, 0.008),
        (1000, 0.0, 0.010),
        (-250, 0.03, 0.010),
        (300, 0.25, 0.040),
    ]  # P Q R S T
    w = int(0.45 * fs)
    tt = np.arange(-w, w + 1) / fs
    template = sum(a * np.exp(-0.5 * ((tt - mu) / sd) ** 2) for a, mu, sd in waves)
    for b in beats:
        lo, hi = max(0, b - w), min(n, b + w + 1)
        ecg[lo:hi] += (1 + 0.06 * breath[b]) * template[lo - (b - w) : hi - (b - w)]
    src["ecg"] = ecg + 40 * _slow(n, fs, rng, 3.0) + _colored(n, rng, 15, 0.3)

    # ---- EEG components (uV)
    arousal = np.zeros(n)
    for ev in study.events:
        arousal = np.maximum(arousal, _box(t, ev.end, ev.end + 4, ramp=0.5))
    alpha = (
        by_stage({"W": 15, "N1": 4, "N2": 2, "N3": 1, "R": 4}) + 20 * arousal
    ) * _osc(t, 10, rng, n, fs)
    theta = by_stage({"W": 4, "N1": 12, "N2": 8, "N3": 6, "R": 12}) * _osc(
        t, 6, rng, n, fs
    )
    delta = by_stage({"W": 4, "N1": 8, "N2": 18, "N3": 55, "R": 6}) * _osc(
        t, 1.5, rng, n, fs
    )
    spindle = np.zeros(n)
    for s in np.arange(5, t[-1] - 2, 9.0) + rng.uniform(
        -3, 3, len(np.arange(5, t[-1] - 2, 9.0))
    ):
        i0 = int(s * fs)
        if stage[i0] == "N2":
            seg = np.arange(fs) / fs
            spindle[i0 : i0 + fs] += 15 * np.hanning(fs) * np.sin(2 * np.pi * 13 * seg)

    def eeg():
        g = rng.uniform(0.7, 1.3, 4)
        return (
            g[0] * alpha
            + g[1] * theta
            + g[2] * delta
            + g[3] * spindle
            + _colored(n, rng, 6)
        )

    src["eeg"] = {k: eeg() for k in ("F3", "F4", "C3", "C4", "O1", "O2", "CZ")}

    # ---- EOG (uV): blinks in wake, slow rolling in N1, rapid movements in REM
    rem_impulses = (
        (rng.random(n) < 0.6 / fs)
        * (stage == "R")
        * rng.choice([-1, 1], n)
        * rng.uniform(60, 120, n)
    )
    rem = sps.lfilter([1.0], [1.0, -np.exp(-1 / (0.4 * fs))], rem_impulses)
    rolling = 40 * (stage == "N1") * np.sin(2 * np.pi * 0.25 * t)
    blink_imp = (rng.random(n) < 0.25 / fs) * (stage == "W") * 150.0
    blink = np.convolve(
        blink_imp,
        np.exp(-0.5 * (np.arange(-fs // 2, fs // 2) / (0.08 * fs)) ** 2),
        "same",
    )
    src["eog_l"] = rem + rolling + blink + 0.3 * delta + _colored(n, rng, 8)
    src["eog_r"] = -rem - 0.8 * rolling + blink + 0.3 * delta + _colored(n, rng, 8)

    src["emg"] = by_stage(
        {"W": 20, "N1": 12, "N2": 10, "N3": 8, "R": 3}
    ) * rng.standard_normal(n)
    snore_gate = np.maximum(post, paradox) * np.abs(breath)
    src["snore"] = (5 + 80 * snore_gate) * rng.standard_normal(n)
    return src


def nch_channels(src) -> list[tuple[str, str, np.ndarray]]:
    e = src["eeg"]
    return [
        ("EEG F3-M2", "uV", e["F3"]),
        ("EEG F4-M1", "uV", e["F4"]),
        ("EEG C3-M2", "uV", e["C3"]),
        ("EEG C4-M1", "uV", e["C4"]),
        ("EEG O1-M2", "uV", e["O1"]),
        ("EEG O2-M1", "uV", e["O2"]),
        ("EEG CZ-O1", "uV", e["CZ"]),
        ("EOG LOC-M2", "uV", src["eog_l"]),
        ("EOG ROC-M1", "uV", src["eog_r"]),
        ("EMG Chin1-Chin2", "uV", src["emg"]),
        ("ECG EKG2-EKG", "uV", src["ecg"]),
        ("Resp PTAF", "mV", src["ptaf"]),
        ("Resp Airflow", "mV", src["airflow"]),
        ("Resp Thoracic", "mV", src["thoracic"]),
        ("Resp Abdominal", "mV", src["abdominal"]),
        ("SpO2", "%", src["spo2"]),
        ("Capno", "mmHg", src["capno"]),
        ("Snore", "uV", src["snore"]),
    ]


def chat_channels(src, rng) -> list[tuple[str, str, np.ndarray]]:
    # CHAT stores single-ended electrodes; preprocessing re-references them.
    n = len(src["ecg"])
    m1, m2 = _colored(n, rng, 5), _colored(n, rng, 5)
    e = src["eeg"]
    ecg_common = _colored(n, rng, 40)
    return [
        ("E1", "uV", src["eog_l"] + m2),
        ("E2", "uV", src["eog_r"] + m1),
        ("F3", "uV", e["F3"] + m2),
        ("F4", "uV", e["F4"] + m1),
        ("C3", "uV", e["C3"] + m2),
        ("C4", "uV", e["C4"] + m1),
        ("M1", "uV", m1),
        ("M2", "uV", m2),
        ("O1", "uV", e["O1"] + m2),
        ("O2", "uV", e["O2"] + m1),
        ("ECG1", "uV", 0.5 * src["ecg"] + ecg_common),
        ("ECG3", "uV", -0.5 * src["ecg"] + ecg_common),
        ("Lchin", "uV", src["emg"]),
        ("CannulaFlow", "mV", src["ptaf"]),
        ("Airflow", "mV", src["airflow"]),
        ("Chest", "mV", src["thoracic"]),
        ("ABD", "mV", src["abdominal"]),
        ("SaO2", "%", src["spo2"]),
        ("Cap", "mmHg", src["capno"]),
        ("Snore", "uV", src["snore"]),
    ]


def annotations(study: Study) -> pd.DataFrame:
    nch = study.spec.dataset == "nch"
    stage_names, event_names = (
        (NCH_STAGE, NCH_EVENT) if nch else (CHAT_STAGE, CHAT_EVENT)
    )
    rows = [
        (i * EPOCH_SEC, EPOCH_SEC, stage_names[s]) for i, s in enumerate(study.stages)
    ]
    for ev in study.events:
        rows.append((ev.onset, ev.duration, event_names[ev.kind]))
        rows.append((round(ev.end + 0.5, 1), 3.0, "EEG arousal" if nch else "Arousal"))
        rows.append(
            (
                round(ev.end + 3.0, 1),
                12.0,
                "Oxygen Desaturation" if nch else "SpO2 desaturation",
            )
        )
    if nch:
        rows += [
            (0.0, 0.0, "Lights Off"),
            (float(len(study.stages) * EPOCH_SEC - 5), 0.0, "Lights On"),
        ]
    df = pd.DataFrame(rows, columns=["onset", "duration", "description"])
    return df.sort_values(["onset", "duration"], kind="stable").reset_index(drop=True)


# NSRR XML: EventType and EventConcept are "<short>|<long>" pairs.
_XML_CONCEPT = {
    "Wake": ("Stages|Stages", "Wake|0"),
    "Stage 1 sleep": ("Stages|Stages", "Stage 1 sleep|1"),
    "Stage 2 sleep": ("Stages|Stages", "Stage 2 sleep|2"),
    "Stage 3 sleep": ("Stages|Stages", "Stage 3 sleep|3"),
    "REM sleep": ("Stages|Stages", "REM sleep|5"),
    "Obstructive apnea": (
        "Respiratory|Respiratory",
        "Obstructive apnea|Obstructive Apnea",
    ),
    "Central apnea": ("Respiratory|Respiratory", "Central apnea|Central Apnea"),
    "Hypopnea": ("Respiratory|Respiratory", "Hypopnea|Hypopnea"),
    "SpO2 desaturation": (
        "Respiratory|Respiratory",
        "SpO2 desaturation|SpO2 desaturation",
    ),
    "Arousal": ("Arousals|Arousals", "Arousal|Arousal ()"),
}


def write_nsrr_xml(
    path: Path, df: pd.DataFrame, start: datetime, total_sec: float
) -> None:
    """Write annotations in the NSRR ``*-nsrr.xml`` layout."""
    import xml.etree.ElementTree as ET

    root = ET.Element("PSGAnnotation")
    ET.SubElement(root, "SoftwareVersion").text = "Compumedics"
    ET.SubElement(root, "EpochLength").text = str(EPOCH_SEC)
    events = ET.SubElement(root, "ScoredEvents")
    first = ET.SubElement(events, "ScoredEvent")
    ET.SubElement(first, "EventType")
    ET.SubElement(first, "EventConcept").text = "Recording Start Time"
    ET.SubElement(first, "Start").text = "0"
    ET.SubElement(first, "Duration").text = f"{total_sec:.1f}"
    ET.SubElement(first, "ClockTime").text = f"00.00.00 {start.strftime('%H.%M.%S')}"
    for row in df.itertuples():
        etype, concept = _XML_CONCEPT[row.description]
        ev = ET.SubElement(events, "ScoredEvent")
        ET.SubElement(ev, "EventType").text = etype
        ET.SubElement(ev, "EventConcept").text = concept
        ET.SubElement(ev, "Start").text = f"{row.onset:g}"
        ET.SubElement(ev, "Duration").text = f"{row.duration:.1f}"
        if etype.startswith("Respiratory"):
            ET.SubElement(ev, "SignalLocation").text = (
                "SaO2" if "SpO2" in concept else "Airflow"
            )
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


# =============================================================== stage 1: raw
def _edf_channels(chans, fs: int) -> list[dict]:
    """Turn (label, unit, data) tuples into write_edf channel dicts."""
    return [
        {"label": label, "unit": unit, "fs": fs, "data": data}
        for label, unit, data in chans
    ]


def write_raw(out: Path, minutes: int, seed: int) -> dict[str, list[Study]]:
    n_epochs = minutes * 60 // EPOCH_SEC
    nch_dir = out / "raw" / "nchsdb"
    chat_dir = out / "raw" / "chat"
    (nch_dir / "sleep_data").mkdir(parents=True)
    (nch_dir / "health_data").mkdir(parents=True)
    (chat_dir / "polysomnography" / "edfs" / "baseline").mkdir(parents=True)
    (chat_dir / "datasets").mkdir(parents=True)

    studies: dict[str, list[Study]] = {"nch": [], "chat": []}
    base_date = datetime(2017, 3, 4, 21, 30, 0)
    for k, spec in enumerate(nch_specs() + chat_specs()):
        rng = np.random.default_rng(seed * 1000 + k)
        study = Study(spec, make_hypnogram(n_epochs))
        place_events(study, rng)
        src = synthesize(study, rng)
        chans = nch_channels(src) if spec.dataset == "nch" else chat_channels(src, rng)
        chans = [c for c in chans if c[0] not in spec.drop_channels]
        start = base_date + timedelta(days=37 * k, minutes=7 * k)
        if spec.dataset == "nch":
            stem = nch_dir / "sleep_data" / spec.name
            write_edf(stem.with_suffix(".edf"), _edf_channels(chans, spec.fs), start)
            annotations(study).to_csv(stem.with_suffix(".tsv"), sep="\t", index=False)
        else:
            psg = chat_dir / "polysomnography"
            stem = psg / "edfs" / spec.visit / spec.name
            stem.parent.mkdir(parents=True, exist_ok=True)
            write_edf(stem.with_suffix(".edf"), _edf_channels(chans, spec.fs), start)
            ann = annotations(study)
            write_nsrr_xml(
                psg / "annotations-events-nsrr" / spec.visit / f"{spec.name}-nsrr.xml",
                ann,
                start,
                len(study.stages) * EPOCH_SEC,
            )
            if spec.visit == "baseline":
                # apnea-missing-modality reads a TSV next to the EDF;
                # CHATDataset reads the XML when both are there.
                ann.to_csv(f"{stem}-nsrr.tsv", sep="\t", index=False)
        study.start = start
        studies[spec.dataset].append(study)

    # ---- NCH health data
    nch = studies["nch"]
    pd.DataFrame(
        {
            "STUDY_PAT_ID": [int(s.spec.patient_id) for s in nch],
            "SLEEP_STUDY_ID": [int(s.spec.name.split("_")[1]) for s in nch],
            "SLEEP_STUDY_START_DATETIME": [
                s.start.strftime("%m/%d/%Y %I:%M:%S %p") for s in nch
            ],
            "SLEEP_STUDY_DURATION_DATETIME": [
                f"0 days {minutes // 60:02d}:{minutes % 60:02d}:00" for _ in nch
            ],
            "AGE_AT_SLEEP_STUDY_DAYS": [s.spec.age_days for s in nch],
        }
    ).to_csv(nch_dir / "health_data" / "SLEEP_STUDY.csv", index=False)
    first = {}
    for s in nch:
        first.setdefault(s.spec.patient_id, s)
    pd.DataFrame(
        {
            "STUDY_PAT_ID": [int(p) for p in first],
            "BIRTH_DATE": [
                (s.start - timedelta(days=s.spec.age_days)).strftime("%Y-%m-%d")
                for s in first.values()
            ],
            "GENDER_DESCR": [
                "Male" if s.spec.male else "Female" for s in first.values()
            ],
            "RACE_DESCR": ["Unknown"] * len(first),
            "ETHNICITY_DESCR": ["Unknown"] * len(first),
        }
    ).to_csv(nch_dir / "health_data" / "DEMOGRAPHIC.csv", index=False)

    for visit in ("baseline", "followup"):
        chat = [s for s in studies["chat"] if s.spec.visit == visit]
        pd.DataFrame(
            {
                "nsrrid": [int(s.spec.patient_id) for s in chat],
                "ageyear_at_meas": [round(s.spec.age_days / 365, 1) for s in chat],
                "male": [s.spec.male for s in chat],
                "race3": [1 + i % 3 for i in range(len(chat))],
                "ahi_a0h3": [round(s.ahi, 2) for s in chat],
            }
        ).to_csv(chat_dir / "datasets" / f"chat-{visit}-dataset-mock.csv", index=False)
    return studies


# =============================================================== main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument(
        "--minutes", type=int, default=20, help="recording length per study"
    )
    ap.add_argument("--seed", type=int, default=598)
    ap.add_argument(
        "--force", action="store_true", help="delete --out first if it exists"
    )
    args = ap.parse_args()
    if args.minutes < 5:
        ap.error("--minutes must be >= 5 so every study has sleep epochs")
    if args.out.exists():
        if not args.force:
            ap.error(f"{args.out} exists; pass --force to replace it")
        shutil.rmtree(args.out)

    studies = write_raw(args.out, args.minutes, args.seed)
    manifest = {
        "minutes": args.minutes,
        "seed": args.seed,
        "studies": {
            s.spec.name: {
                "dataset": s.spec.dataset,
                "patient_id": s.spec.patient_id,
                "fs": s.spec.fs,
                "n_events": len(s.events),
                "ahi": round(s.ahi, 2),
                "note": s.spec.note,
                "events": [[e.onset, e.duration, e.kind] for e in s.events],
            }
            for s in studies["nch"] + studies["chat"]
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {args.out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
