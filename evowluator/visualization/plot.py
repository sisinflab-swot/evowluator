from __future__ import annotations

import sys
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt, ticker

from .metric import Metric


class Plot:

    # Override

    def draw(self) -> None:
        self.draw_grid()
        self.draw_legend()
        self.draw_titles()

    # Public

    def __init__(self, ax: plt.Axes):
        self.data = None
        self.grid_axis = 'both'
        self.show_titles = True
        self.title: Optional[str] = None
        self.xlabel: Optional[str] = None
        self.ylabel: Optional[str] = None

        self._ax = ax

        ax.xaxis.set_major_formatter(self._default_formatter())
        ax.yaxis.set_major_formatter(self._default_formatter())

    def set_attrs(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def set_scale(self, scale: str, axis: str = 'both') -> None:
        # Workaround for formatter getting reset on set_[xy]scale.
        x_maj = self._ax.xaxis.get_major_formatter()
        x_min = self._ax.xaxis.get_minor_formatter()
        y_maj = self._ax.yaxis.get_major_formatter()
        y_min = self._ax.yaxis.get_minor_formatter()

        if axis != 'x':
            self._ax.set_yscale(scale)

        if axis != 'y':
            self._ax.set_xscale(scale)

        self._ax.xaxis.set_major_formatter(x_maj)
        self._ax.xaxis.set_minor_formatter(x_min)
        self._ax.yaxis.set_major_formatter(y_maj)
        self._ax.yaxis.set_minor_formatter(y_min)

    def draw_grid(self) -> None:
        self._ax.minorticks_on()
        self._ax.set_axisbelow(True)
        self._ax.grid(b=True, axis=self.grid_axis, which='major')
        self._ax.grid(b=True, axis=self.grid_axis, which='minor', alpha=0.25)

    def draw_legend(self) -> None:
        legend = self._ax.legend()
        legend.set_draggable(True)

    def draw_titles(self) -> None:
        if not self.show_titles:
            return

        if self.title:
            self._ax.set_title(self.title)

        if self.xlabel:
            self._ax.set_xlabel(self.xlabel)

        if self.ylabel:
            self._ax.set_ylabel(self.ylabel)

    def _default_formatter(self) -> ticker.Formatter:
        return ticker.FormatStrFormatter('%g')


class HistogramPlot(Plot):

    def draw(self) -> None:
        self.draw_labels()
        super().draw()

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.center_labels = False
        self.data: Dict[str, List[float]] = {}
        self.grid_axis = 'y'
        self.label_fmt: Optional[str] = None
        self.metric: Optional[Metric] = None
        self.show_labels = True

    def draw_labels(self) -> None:
        if not self.show_labels:
            return

        fmt = self.label_fmt if self.label_fmt else self.metric.fmt

        if not fmt:
            return

        fig = self._ax.get_figure()
        transform = fig.dpi_scale_trans.inverted()

        x_mult = 0.5

        for rect in self._ax.patches:
            height_px = rect.get_bbox().transformed(transform).height * fig.dpi

            w = rect.get_width()
            h = rect.get_height()

            if self.center_labels and height_px > 20.0:
                y_mult = 0.5
                va = 'center'
            else:
                y_mult = 1.0
                va = 'bottom'

            x = rect.get_x() + w * x_mult
            y = rect.get_y() + h * y_mult

            annotation = self._ax.annotate(format(h, fmt), (x, y), ha='center', va=va)
            annotation.draggable()

    def configure(self, data_min: float, data_max: float) -> None:
        if data_min == 0.0:
            data_min = sys.float_info.epsilon

        if data_max / data_min > 25.0:
            self.set_scale('log' if data_min > 1.0 else 'symlog', axis='y')
            bottom, top = self.ylim_log_scale(data_min, data_max)
        else:
            bottom, top = self.ylim_linear_scale(data_min, data_max)

        self._ax.set_ylim(bottom=bottom, top=top)

    def ylim_log_scale(self, data_min: float, data_max: float) -> (float, float):
        bottom = 10.0 ** np.floor(np.log10(data_min))
        top = 10.0 ** np.ceil(np.log10(data_max))

        if data_max / top > 0.65:
            top *= 2.0
        elif data_max / top < 0.2:
            top *= 0.3

        return bottom, top

    def ylim_linear_scale(self, data_min: float, data_max: float) -> (float, float):
        mult = 10.0 ** np.floor(np.log10(data_max))

        bottom = (data_min // mult) * mult
        top = (data_max // mult + 1.0) * mult

        if top - data_max < mult * 0.4:
            top += mult * 0.5

        if data_min - bottom < mult * 0.4:
            bottom = max(bottom - mult, 0.0)

        return bottom, top


class GroupedHistogramPlot(HistogramPlot):

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.groups: List[str] = []

    def draw(self) -> None:
        data_min = min(p for l in self.data.values() for p in l)
        data_max = max(p for l in self.data.values() for p in l)
        self.configure(data_min, data_max)

        labels = list(self.data.keys())
        labels.sort()

        n_labels = len(labels)
        n_groups = len(self.groups)

        width = 1.0 / (n_labels + 1)
        bar_width = 0.8 * width

        for i, label in enumerate(labels):
            self._ax.bar([j + width * i for j in range(n_groups)], self.data[label],
                         width=bar_width, alpha=0.9, label=label)

        self._ax.set_xticks([p + width * ((n_labels - 1) / 2) for p in range(n_groups)])
        self._ax.set_xticklabels(self.groups)

        self.ylabel = self.metric.to_string(capitalize=True)
        self.title = self.metric.capitalized_name

        super().draw()


class MinMaxAvgHistogramPlot(GroupedHistogramPlot):

    def draw(self) -> None:
        self.title = 'Minimum, average and maximum {}'.format(self.metric.name)
        self.groups = ['Min', 'Avg', 'Max']
        super().draw()


class StackedHistogramPlot(HistogramPlot):

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.data: Dict[str, List[float]] = {}
        self.labels: List[str] = []

    def draw(self) -> None:
        data_min = min(p for l in self.data.values() for p in l)
        data_max = max(sum(l) for l in self.data.values())
        self.configure(data_min, data_max)

        group_labels = list(self.data.keys())
        group_labels.sort()
        n_group_labels = len(group_labels)

        n_sections = len(next(iter(self.data.values())))
        pos = np.arange(n_group_labels)
        width = 0.5

        values = [self.data[r][0] for r in group_labels]
        self._ax.bar(pos, values, width, alpha=0.9, label=self.labels[0])

        for section in range(1, n_sections):
            prev_values = values
            values = [self.data[r][section] for r in group_labels]
            self._ax.bar(pos, values, width, alpha=0.9, bottom=prev_values,
                         label=self.labels[section])

        self._ax.set_xticks(pos)
        self._ax.set_xticklabels(group_labels)

        super().draw()


class ScatterPlot(Plot):

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.data: Dict[str, Tuple[List[float], List[float]]] = {}
        self.xmetric: Optional[Metric] = None
        self.ymetric: Optional[Metric] = None

    def draw(self, **kwargs) -> None:
        labels = list(self.data.keys())
        labels.sort()

        dataset_size = len(next(iter(self.data.values()))[0])
        point_size = self.configure(dataset_size)

        for label in labels:
            x, y = self.data[label]
            self._ax.scatter(x, y, s=point_size, alpha=0.5, label=label)
            self.draw_polyline(x, y)

        self.title = '{} by {}'.format(self.ymetric.capitalized_name, self.xmetric.name)
        self.xlabel = self.xmetric.to_string(capitalize=True)
        self.ylabel = self.ymetric.to_string(capitalize=True)

        super().draw()

    def draw_polyline(self, x: List[float], y: List[float]) -> None:
        count = len(x)
        weights = [1.0] * count

        # Force start from first data points
        count = max(count // (100 if count > 100 else 5), 1)
        y[0] = sum(y[:count]) / count
        weights[0] = max(y) * 10.0

        self._ax.plot(x, np.poly1d(np.polyfit(x, y, 1, w=weights))(x))

    def configure(self, dataset_size: int) -> float:
        if dataset_size > 100:
            self.set_scale('log')
            point_size = 10.0
        else:
            point_size = 50.0

        return point_size


class Plotter:

    def __init__(self, plot_type: type, **kwargs):
        self._plot_type = plot_type
        self._args = kwargs

    def draw_plot(self, ax: plt.Axes) -> None:
        plot = self._plot_type(ax)
        plot.set_attrs(**self._args)
        plot.draw()


class Figure:

    def __init__(self):
        self.title = 'evOWLuator - results'
        self.size: Optional[Tuple[float, float]] = None
        self.show_titles = True
        self.show_labels = True
        self.label_fmt: Optional[str] = None
        self._plotters: List[Plotter] = []

    def add_plotter(self, plot_type: type, **kwargs) -> None:
        if self.label_fmt:
            kwargs['label_fmt'] = self.label_fmt

        kwargs['show_titles'] = self.show_titles
        kwargs['show_labels'] = self.show_labels

        self._plotters.append(Plotter(plot_type, **kwargs))

    def show(self, plots: Optional[List[int]] = None) -> None:
        plotters = [self._plotters[i] for i in plots] if plots else self._plotters

        n_plots = len(plotters)
        n_rows = 2 if n_plots > 1 else 1
        n_cols = ceil(n_plots / n_rows)
        n_figures = n_rows * n_cols

        if not self.size:
            height = 5 if n_figures == 1 else 9
            width = 10 if n_figures <= 2 else 16
            self.size = (width, height)

        fig, axes = plt.subplots(figsize=self.size, nrows=n_rows, ncols=n_cols, squeeze=False)
        fig.canvas.set_window_title(self.title)
        axes = axes.flatten()

        for i, plotter in enumerate(plotters):
            plotter.draw_plot(axes[i])

        fig.tight_layout()
        plt.show()
