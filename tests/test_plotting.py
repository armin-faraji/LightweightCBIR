from __future__ import annotations

import unittest

from cbir.plotting import SeriesData, plot_series


class PlottingTests(unittest.TestCase):
    def test_legend_title_describes_coloured_series(self) -> None:
        figure, axis = plot_series(
            {
                "0.05": SeriesData(x=[0, 1], y=[0.8, 0.7]),
                "0.1": SeriesData(x=[0, 1], y=[0.9, 0.8]),
            },
            legend_title="Pooling temperature (τₚ)",
        )
        try:
            legend = axis.get_legend()
            self.assertIsNotNone(legend)
            self.assertEqual(legend.get_title().get_text(), "Pooling temperature (τₚ)")
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
