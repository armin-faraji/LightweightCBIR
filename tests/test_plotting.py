from __future__ import annotations

import unittest

from cbir.plotting import HorizontalReference, SeriesData, plot_series


class PlottingTests(unittest.TestCase):
    def _series(self) -> dict[str, SeriesData]:
        return {"run": SeriesData(x=[0, 1], y=[0.8, 0.7])}

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

    def test_default_figure_size_is_backward_compatible(self) -> None:
        figure, _ = plot_series(self._series())
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (7.0, 4.5))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_custom_figure_size_is_used_for_new_figure(self) -> None:
        figure, _ = plot_series(self._series(), fig_size=(12, 6))
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (12.0, 6.0))
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_invalid_figure_size_is_rejected(self) -> None:
        for value in ((0, 4), (float("nan"), 4), (4,), "4, 5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fig_size"):
                    plot_series(self._series(), fig_size=value)  # type: ignore[arg-type]

    def test_figure_size_cannot_be_used_with_existing_axis(self) -> None:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "together with ax"):
                plot_series(self._series(), ax=axis, fig_size=(12, 6))
        finally:
            plt.close(figure)

    def test_horizontal_reference_is_in_the_legend(self) -> None:
        figure, axis = plot_series(
            self._series(),
            horizontal_references={
                "Final CLS — 384-D (frozen)": HorizontalReference(0.92),
            },
        )
        try:
            self.assertEqual(len(axis.lines), 2)
            legend = axis.get_legend()
            self.assertIsNotNone(legend)
            labels = [text.get_text() for text in legend.get_texts()]
            self.assertIn("Final CLS — 384-D (frozen)", labels)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
