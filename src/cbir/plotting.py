"""Reusable plotting helpers for experiments and report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class HorizontalReference:
    """A labelled horizontal reference line for a plot."""

    value: float
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.5

    def __post_init__(self) -> None:
        if not isfinite(float(self.value)):
            raise ValueError("horizontal reference value must be finite")
        if not isfinite(float(self.linewidth)) or self.linewidth <= 0:
            raise ValueError("horizontal reference linewidth must be positive")


def plot_series(
    series: Mapping[str, SeriesData],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colors: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    horizontal_references: Mapping[str, HorizontalReference] | None = None,
    legend_title: str | None = None,
    error_bars: bool = True,
    fig_size: Sequence[float] | None = None,
    ax: object | None = None,
    save_path: Path | None = None,
    save_dpi: int = 300,
) -> tuple[object, object]:
    """Plot arbitrary named series with stable colors, legends, and error support.

    ``legend_title`` explains what the named, coloured series represent.  For
    example, use ``"Pooling temperature (τₚ)"`` when each series is one
    temperature value, rather than leaving a reader to infer that from labels
    such as ``0.05`` and ``0.1``.

    ``fig_size`` supplies the ``(width, height)`` in inches for a newly created
    figure.  It is deliberately unavailable together with ``ax`` because that
    figure has already been created and sized by the caller.  Omitting it keeps
    the established ``(7, 4.5)`` default, so existing notebook calls remain
    visually and programmatically unchanged.

    When ``save_path`` is supplied, the default 300 DPI is suitable for inserting
    the generated PNG directly into a course report.  Pass a lower ``save_dpi``
    only for disposable diagnostic plots.
    """
    if not series:
        raise ValueError("at least one named series is required")
    if save_dpi <= 0:
        raise ValueError("save_dpi must be positive")
    if ax is not None and fig_size is not None:
        raise ValueError("fig_size cannot be used together with ax")
    resolved_fig_size = _validate_fig_size(fig_size)
    if ax is None:
        figure, axis = plt.subplots(figsize=resolved_fig_size)
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
    for name, reference in (horizontal_references or {}).items():
        axis.axhline(
            reference.value,
            label=name,
            color=reference.color,
            linestyle=reference.linestyle,
            linewidth=reference.linewidth,
        )
    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(title=legend_title)
    figure.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    return figure, axis


def _validate_fig_size(fig_size: Sequence[float] | None) -> tuple[float, float]:
    """Return a safe Matplotlib figure size while preserving the old default."""
    if fig_size is None:
        return (7.0, 4.5)
    if isinstance(fig_size, (str, bytes)):
        raise ValueError("fig_size must be a (width, height) pair in inches")
    try:
        if len(fig_size) != 2:
            raise ValueError("fig_size must be a (width, height) pair in inches")
    except TypeError as error:
        raise ValueError("fig_size must be a (width, height) pair in inches") from error
    try:
        width, height = (float(value) for value in fig_size)
    except (TypeError, ValueError) as error:
        raise ValueError("fig_size must contain numeric width and height values") from error
    if not isfinite(width) or not isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("fig_size values must be finite and positive")
    return (width, height)
