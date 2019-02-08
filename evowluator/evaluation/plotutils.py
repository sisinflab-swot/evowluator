from typing import List, Union

from matplotlib import pyplot as plt, ticker


def setup_plot(**kwargs) -> (plt.Figure, Union[plt.Axes, List[plt.Axes]]):
    if kwargs.get('figsize') is None:
        kwargs['figsize'] = (10, 10)

    kwargs['squeeze'] = False
    fig, axes = plt.subplots(**kwargs)
    fig.canvas.set_window_title('evOWLuator - results')

    for row in axes:
        for ax in row:
            ax.xaxis.set_major_formatter(default_formatter())
            ax.yaxis.set_major_formatter(default_formatter())

    return fig, axes


def set_scale(ax: plt.Axes, scale: str, axis: str = 'both') -> None:
    """Workaround for formatter getting reset on set_[xy]scale."""
    x_maj = ax.xaxis.get_major_formatter()
    x_min = ax.xaxis.get_minor_formatter()
    y_maj = ax.yaxis.get_major_formatter()
    y_min = ax.yaxis.get_minor_formatter()

    if axis != 'x':
        ax.set_yscale(scale)

    if axis != 'y':
        ax.set_xscale(scale)

    ax.xaxis.set_major_formatter(x_maj)
    ax.xaxis.set_minor_formatter(x_min)
    ax.yaxis.set_major_formatter(y_maj)
    ax.yaxis.set_minor_formatter(y_min)


def default_formatter() -> ticker.Formatter:
    return ticker.FormatStrFormatter('%g')


def configure_scatter_plot(ax: plt.Axes, dataset_size: int) -> float:
    if dataset_size > 100:
        set_scale(ax, 'log')
        point_size = 10.0
    else:
        point_size = 50.0

    return point_size


def display_labels(ax: plt.Axes, center: bool = False, fmt: str = '{:.0f}') -> None:
    x_mult = 0.5

    for rect in ax.patches:
        h = rect.get_height()
        w = rect.get_width()

        if center and h > 40.0:
            y_mult = 0.5
            y_add = -15.0
        else:
            y_mult = 1.05
            y_add = 0.0

        x = rect.get_x() + w * x_mult
        y = rect.get_y() + h * y_mult + y_add

        ax.text(x, y, fmt.format(h), ha='center', va='bottom')


def display_grid(ax: plt.Axes, axis: str = 'both') -> None:
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.grid(b=True, axis=axis, which='major')
    ax.grid(b=True, axis=axis, which='minor', alpha=0.25)
