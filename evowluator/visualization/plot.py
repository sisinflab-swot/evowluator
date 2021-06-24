from __future__ import annotations

import sys
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from evowluator.util.strenum import StrEnum
from .metric import Metric


class LegendLocation(StrEnum):
    NONE = 'none'
    BEST = 'best'
    UPPER_RIGHT = 'upper right'
    UPPER_LEFT = 'upper left'
    LOWER_LEFT = 'lower left'
    LOWER_RIGHT = 'lower right'
    CENTER_LEFT = 'center left'
    CENTER_RIGHT = 'center right'
    LOWER_CENTER = 'lower center'
    UPPER_CENTER = 'upper center'
    CENTER = 'center'


class Plot:

    # Override

    def draw_plot(self) -> None:
        for label in self._ax.get_xticklabels():
            label.set_rotation(self.xtick_rot)

        for label in self._ax.get_yticklabels():
            label.set_rotation(self.ytick_rot)

    # Public

    def __init__(self, ax: plt.Axes):
        self.data = None
        self.colors: Dict[str, str] = {}
        self.grid_axis = 'both'
        self.legend_loc = LegendLocation.BEST
        self.legend_cols = 1
        self.legend_only = False
        self.show_titles = True
        self.title: Optional[str] = None
        self.xlabel: Optional[str] = None
        self.ylabel: Optional[str] = None
        self.xtick_rot = 0.0
        self.ytick_rot = 0.0

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

        subticks = [2, 3, 4, 5, 6, 7, 8, 9]

        if axis != 'x':
            self._ax.set_yscale(scale, subs=subticks)

        if axis != 'y':
            self._ax.set_xscale(scale, subs=subticks)

        self._ax.xaxis.set_major_formatter(x_maj)
        self._ax.xaxis.set_minor_formatter(x_min)
        self._ax.yaxis.set_major_formatter(y_maj)
        self._ax.yaxis.set_minor_formatter(y_min)

    def draw(self) -> None:
        if self.legend_only:
            self.draw_legend_only()
            return

        self.draw_plot()
        self.draw_grid()

        if self.legend_loc != LegendLocation.NONE:
            self.draw_legend()

        if self.show_titles:
            self.draw_titles()

    def draw_legend_only(self) -> None:
        self.draw_plot()
        legend = self.draw_legend()

        for artist in self._ax.get_children():
            if artist is not legend:
                artist.set_visible(False)

        self._ax.relim(visible_only=True)
        self._ax.set_axis_off()

    def draw_grid(self) -> None:
        self._ax.minorticks_on()
        self._ax.set_axisbelow(True)
        self._ax.grid(b=True, axis=self.grid_axis, which='major')
        self._ax.grid(b=True, axis=self.grid_axis, which='minor', alpha=0.25)

    def draw_legend(self):
        legend = self._ax.legend(loc=self.legend_loc.value,
                                 mode='expand' if self.legend_only else None,
                                 ncol=self.legend_cols,
                                 handletextpad=0.4,
                                 handlelength=0.7,
                                 labelspacing=0.25,
                                 columnspacing=1.0,
                                 borderaxespad=0.0 if self.legend_only else None)
        legend.set_draggable(True)
        return legend

    def draw_titles(self) -> None:
        if self.title:
            self._ax.set_title(self.title)

        if self.xlabel:
            self._ax.set_xlabel(self.xlabel)

        if self.ylabel:
            self._ax.set_ylabel(self.ylabel)

    def _default_formatter(self) -> ticker.Formatter:
        return ticker.FormatStrFormatter('%g')


class HistogramPlot(Plot):

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.data: Dict[str, List[float]] = {}
        self.grid_axis = 'y'
        self.label_fmt: Optional[str] = None
        self.label_rot = 0.0
        self.metric: Optional[Metric] = None
        self.show_labels = True
        self._labels: List[plt.Annotation] = []

    def draw_plot(self) -> None:
        if self.show_labels:
            self.draw_labels()
        super().draw_plot()

    def draw_labels(self) -> None:
        fmt = self.label_fmt if self.label_fmt else self.metric.fmt

        if not fmt:
            return

        for rect in (p for p in self._ax.patches if p.get_height() > 0.0):
            self._labels.append(self.draw_label(rect, fmt))

        self.fit_labels()

    def draw_label(self, bar: Rectangle, fmt: str) -> plt.Annotation:
        w, h = bar.get_width(), bar.get_height()

        x = bar.get_x() + w * 0.5
        y = bar.get_y() + h

        label = self._ax.annotate(format(h, fmt), (x, y),
                                  xytext=(0.0, 4.0), textcoords='offset points',
                                  ha='center', va='bottom')
        label.draggable()
        label.set_rotation(self.label_rot)

        return label

    def fit_labels(self) -> None:
        ymin, ymax = float('inf'), -float('inf')
        renderer = self._ax.figure.canvas.get_renderer()

        for box in (label.get_window_extent(renderer=renderer) for label in self._labels):
            if box.ymin < ymin:
                ymin = box.ymin

            if box.ymax > ymax:
                ymax = box.ymax

        transform = self._ax.transData.inverted()
        ymin = transform.transform_point((0, ymin))[1]
        ymax = transform.transform_point((0, ymax))[1]

        # Zero labels are not plotted, in which case ymin is larger than
        # the actual required minimum (zero).
        ymin = min(self._ax.get_ylim()[0], ymin)

        self.configure_limits(ymin, ymax)

    def artist_overlaps_labels(self, artist: plt.Artist) -> bool:
        renderer = self._ax.figure.canvas.get_renderer()
        artist_box = artist.get_window_extent(renderer)

        for box in (label.get_window_extent(renderer) for label in self._labels):
            if box.overlaps(artist_box):
                return True

        return False

    def configure_limits(self, data_min: float, data_max: float) -> None:
        if 'log' in self._ax.get_yscale():
            bottom, top = self.ylim_log_scale(data_min, data_max)
        else:
            bottom, top = self.ylim_linear_scale(data_min, data_max)

        self._ax.set_ylim(bottom=bottom, top=top)

    def configure_scale(self, data_min: float, data_max: float) -> None:
        if data_min == 0.0 or data_max / data_min > 25.0:
            self.set_scale('log' if data_min > 1.0 else 'symlog', axis='y')

    def ylim_log_scale(self, data_min: float, data_max: float) -> (float, float):
        if data_min == 0.0:
            data_min = sys.float_info.epsilon

        if data_max == 0.0:
            data_max = sys.float_info.epsilon

        bottom = 10.0 ** np.floor(np.log10(data_min))
        top = 10.0 ** np.ceil(np.log10(data_max))

        if data_max / top > 0.9:
            top *= 2.0
        elif data_max / top < 0.2:
            top *= 0.2

        return bottom, top

    def ylim_linear_scale(self, data_min: float, data_max: float) -> (float, float):
        mult = 10.0 ** np.floor(np.log10(data_max - data_min))

        bottom = (data_min // mult) * mult
        top = (data_max // mult + 1.0) * mult

        if top - data_max < mult * 0.1:
            top += mult * 0.1

        if data_min - bottom < mult * 0.2:
            bottom = max(bottom - mult, 0.0)

        return bottom, top


class GroupedHistogramPlot(HistogramPlot):

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.groups: List[str] = []

    def draw_plot(self) -> None:
        data_min = min(p for v in self.data.values() for p in v)
        data_max = max(p for v in self.data.values() for p in v)
        self.configure_scale(data_min, data_max)
        self.configure_limits(data_min, data_max)

        labels = list(self.data.keys())
        n_labels = len(labels)
        n_groups = len(self.groups)

        width = 1.0 / (n_labels + 1)
        bar_width = 0.8 * width

        for i, label in enumerate(labels):
            color = self.colors.get(label)
            self._ax.bar([j + width * i for j in range(n_groups)], self.data[label],
                         width=bar_width, alpha=0.9, label=label, color=color)

        self._ax.set_xticks([p + width * ((n_labels - 1) / 2) for p in range(n_groups)])
        self._ax.set_xticklabels(self.groups)

        self.ylabel = self.metric.to_string(capitalize=True)
        self.title = self.metric.capitalized_name
        super().draw_plot()


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

    def draw_plot(self) -> None:
        data_min = min(p for v in self.data.values() for p in v)
        data_max = max(sum(v) for v in self.data.values())
        self.configure_scale(data_min, data_max)
        self.configure_limits(data_min, data_max)

        group_labels = list(self.data.keys())
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
        super().draw_plot()


class ScatterPlot(Plot):

    def __init__(self, ax: plt.Axes):
        super().__init__(ax)
        self.data: Dict[str, Tuple[List[float], List[float]]] = {}
        self.markers: Dict[str, str] = {}
        self.linestyles: Dict[str, str] = {}
        self.marker_size = 0.0
        self.xmetric: Optional[Metric] = None
        self.ymetric: Optional[Metric] = None
        self.legend_handles: Dict[str, Line2D] = {}

    def draw_plot(self) -> None:
        labels = list(self.data.keys())

        dataset_size = len(next(iter(self.data.values()))[0])

        if self.marker_size:
            msize = self.marker_size
        else:
            msize = 3.0 if dataset_size > 100 else 7.0

        xmin = min(p for t in self.data.values() for p in t[0])
        xmax = max(p for t in self.data.values() for p in t[0])
        ymin = min(p for t in self.data.values() for p in t[1])
        ymax = max(p for t in self.data.values() for p in t[1])

        self.configure_scale(xmin, xmax, ymin, ymax)

        for label in labels:
            x, y = self.data[label]
            marker, color, linestyle = self.markers.get(label, 'o'), self.colors.get(label), self.linestyles.get(label)
            lines = self._ax.plot(x, y, linestyle='none', alpha=0.5,
                                  ms=msize, label=label, marker=marker, c=color, lw=1.0)

            if lines:
                self.draw_polyline(x, y, color=lines[0].get_color(), linestyle=linestyle)
                self.legend_handles[label] = Line2D([], [], 
                    linestyle=linestyle, ms=msize, marker=lines[0].get_marker(), color=lines[0].get_color(), alpha=1.0, lw=1.5, label=label)
                

        self.title = '{} by {}'.format(self.ymetric.capitalized_name, self.xmetric.name)
        self.xlabel = self.xmetric.to_string(capitalize=True)
        self.ylabel = self.ymetric.to_string(capitalize=True)
        super().draw_plot()

    def draw_polyline(self, x: List[float], y: List[float], color: Optional[str] = None, linestyle: Optional[Union[str, tuple]] = None) -> None:
        count = len(x)
        weights = [1.0] * count

        # Force start from first data points
        count = max(count // (100 if count > 100 else 5), 1)
        y[0] = sum(y[:count]) / count
        weights[0] = max(y) * 10.0

        self._ax.plot(x, np.poly1d(np.polyfit(x, y, 1, w=weights))(x), color=color, linestyle=linestyle)

    def draw_legend(self):
        legend = self._ax.legend(handles=list(self.legend_handles.values()),
                                 loc=self.legend_loc.value,
                                 mode='expand' if self.legend_only else None,
                                 ncol=self.legend_cols,
                                 handletextpad=0.4,
                                 handlelength=2.5,
                                 labelspacing=0.25,
                                 columnspacing=1.0,
                                 borderaxespad=0.0 if self.legend_only else None)
        legend.set_draggable(True)
        return legend

    def configure_scale(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        xlog = xmax / xmin > 25.0
        ylog = ymax / ymin > 25.0

        if xlog and ylog:
            axis = 'both'
        elif xlog:
            axis = 'x'
        elif ylog:
            axis = 'y'
        else:
            axis = None

        if axis:
            self.set_scale('log', axis=axis)


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
        self.label_rot = 0.0
        self.legend_loc = LegendLocation.BEST
        self.legend_cols = 1
        self.legend_only = True
        self.marker_size = 0.0
        self.xtick_rot = 0.0
        self.ytick_rot = 0.0
        self._plotters: List[Plotter] = []
        self._is_drawn = False

    def add_plotter(self, plot_type: type, **kwargs) -> None:
        if self.label_fmt:
            kwargs['label_fmt'] = self.label_fmt

        kwargs['show_titles'] = self.show_titles
        kwargs['show_labels'] = self.show_labels
        kwargs['legend_loc'] = self.legend_loc
        kwargs['legend_cols'] = self.legend_cols
        kwargs['legend_only'] = self.legend_only
        kwargs['label_rot'] = self.label_rot
        kwargs['xtick_rot'] = self.xtick_rot
        kwargs['ytick_rot'] = self.ytick_rot
        kwargs['marker_size'] = self.marker_size

        self._plotters.append(Plotter(plot_type, **kwargs))

    def draw(self, plots: Optional[List[int]] = None) -> None:
        if self._is_drawn:
            return

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
        self._is_drawn = True

    def show(self) -> None:
        if self._is_drawn:
            plt.show()

    def save(self, path: str) -> None:
        if self._is_drawn:
            plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
