"""Reusable plotting helpers for experiments and report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SeriesData:
    x: Sequence[float]
    y: Sequence[float]
    yerr: Sequence[float] | None = None
    lower: Sequence[float] | None = None
    upper: Sequence[float] | None = None

    def __post_init__(self) -> None:
        length = len(self.x)
        if len(self.y) != length:
            raise ValueError("x and y lengths differ")
        if self.yerr is not None and len(self.yerr) != length:
            raise ValueError("yerr length differs from x")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper confidence bands must be supplied together")
        if self.lower is not None and (
            len(self.lower) != length or len(self.upper or ()) != length
        ):
            raise ValueError("confidence band lengths differ from x")


def plot_series(
    series: Mapping[str, SeriesData],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    error_bars: bool = True,
    ax: object | None = None,
    save_path: Path | None = None,
    save_dpi: int = 300,
) -> tuple[object, object]:
    """Plot arbitrary named series with stable colors, legends, and error support.

    When ``save_path`` is supplied, the default 300 DPI is suitable for inserting
    the generated PNG directly into a course report.  Pass a lower ``save_dpi``
    only for disposable diagnostic plots.
    """
    if not series:
        raise ValueError("at least one named series is required")
    if save_dpi <= 0:
        raise ValueError("save_dpi must be positive")
    import matplotlib.pyplot as plt

    if ax is None:
        figure, axis = plt.subplots(figsize=(7, 4.5))
    else:
        axis = ax
        figure = axis.figure
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for position, (name, values) in enumerate(series.items()):
        color = (colors or {}).get(name, palette[position % len(palette)])
        marker = (markers or {}).get(name, "o")
        kwargs = {"label": name, "color": color, "marker": marker}
        if error_bars and values.yerr is not None:
            axis.errorbar(values.x, values.y, yerr=values.yerr, capsize=3, **kwargs)
        else:
            axis.plot(values.x, values.y, **kwargs)
        if values.lower is not None and values.upper is not None:
            axis.fill_between(
                values.x,
                np.asarray(values.lower),
                np.asarray(values.upper),
                color=color,
                alpha=0.18,
            )
    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    return figure, axis
