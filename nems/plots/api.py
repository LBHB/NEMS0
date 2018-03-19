from .assemble import (simple_grid, freeze_defaults, get_predictions,
                       plot_layout, combine_signal_channels, pad_to_signals)
from .scatter import plot_scatter
from .summary import plot_summary
from .spectrogram import (plot_spectrogram, spectrogram_from_signal,
                          spectrogram_from_epoch)
from .timeseries import (timeseries_from_signals, timeseries_from_epoch,
                         before_and_after)
from .heatmap import weight_channels_heatmap, fir_heatmap, strf_heatmap
from .file import save_figure, load_figure_img, load_figure_bytes, fig2BytesIO
from .histogram import pred_error_hist
