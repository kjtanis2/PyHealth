"""Small synthetic sleep study files for the NCH, CHAT and apnea task tests."""

from pathlib import Path

import numpy as np
import pandas as pd


def write_edf(
    path: Path,
    labels,
    sfreq: int = 8,
    seconds: int = 120,
    signals: dict | None = None,
    units: dict | None = None,
) -> None:
    """Write a 16-bit EDF with one-second records, readable by MNE.

    MNE cannot write EDF without the optional ``edfio`` package, so the
    header is written by hand.

    Args:
        path: Output file.
        labels: Channel labels.
        sfreq: Rate of every channel in Hz.
        seconds: Recording length.
        signals: Optional label -> values (``seconds * sfreq`` of them, in
            the channel's unit). Channels not given are zeros.
        units: Optional label -> physical dimension. Defaults to ``uV``.
    """

    def field(value, width):
        return str(value).ljust(width).encode("ascii")

    signals = signals or {}
    units = units or {}
    ns = len(labels)
    n = seconds * sfreq
    data = [
        np.asarray(signals.get(label, np.zeros(n)), dtype=float) for label in labels
    ]
    low = [min(float(d.min()), -100.0) for d in data]
    high = [max(float(d.max()), 100.0) for d in data]
    header = b"".join(
        [
            field(0, 8),
            field("X X X X", 80),
            field("Startdate 04-MAR-2017 X X X", 80),
            field("04.03.17", 8),
            field("21.30.00", 8),
            field(256 * (ns + 1), 8),
            field("", 44),
            field(seconds, 8),
            field(1, 8),
            field(ns, 4),
        ]
    )
    for width, values in (
        (16, labels),
        (80, [""] * ns),
        (8, [units.get(label, "uV") for label in labels]),
        (8, [f"{v:.6g}" for v in low]),
        (8, [f"{v:.6g}" for v in high]),
        (8, [-32768] * ns),
        (8, [32767] * ns),
        (80, [""] * ns),
        (8, [sfreq] * ns),
        (32, [""] * ns),
    ):
        header += b"".join(field(v, width) for v in values)
    digital = np.stack(
        [
            np.round((d - lo) / (hi - lo) * 65535 - 32768)
            for d, lo, hi in zip(data, low, high)
        ]
    ).astype("<i2")
    # One record per second: each channel's samples of that second in turn.
    records = digital.reshape(ns, seconds, sfreq).transpose(1, 0, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + records.tobytes())


def write_tsv(path: Path, rows) -> None:
    """Write ``(onset, duration, description)`` rows as a tab-separated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["onset", "duration", "description"]).to_csv(
        path, sep="\t", index=False
    )
