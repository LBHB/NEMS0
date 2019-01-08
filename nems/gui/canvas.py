import matplotlib.ticker as tkr
import numpy as np
import PyQt5.QtWidgets as qw
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import nems.signal
from nems.plots.utils import ax_remove_box


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_position([0.175, 0.175, 0.775, 0.7])

        super(FigureCanvas, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, qw.QSizePolicy.Expanding,
                                   qw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class NemsCanvas(MyMplCanvas):

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        if 'mask' in recording.signals:
            self.recording = recording.apply_mask()
        else:
            self.recording = recording
        self.signal = signal
        self.signal_obj = self.recording[self.signal]
        self.fs = self.signal_obj.fs
        self.parent = parent
        print("creating canvas: {}".format(signal))

        sig_array = self.signal_obj.as_continuous()
        # Chop off end of array (where it's all nan'd out after processing)
        # TODO: Make this smarter incase there are intermediate nans?
        self.max_time = sig_array.shape[-1] / self.recording[self.signal].fs

        point = (isinstance(self.recording[self.signal],
                            nems.signal.PointProcess))
        tiled = (isinstance(self.recording[self.signal],
                            nems.signal.TiledSignal)
                 or 'stim' in self.recording[self.signal].name
                 or 'contrast' in self.recording[self.signal].name)

        if (not point) and (not tiled):
            self.ymax = np.nanmax(sig_array)*1.25
            self.ymin = min(0, np.nanmin(sig_array)*1.25)

        self.point = point
        self.tiled = tiled

        # skip some channels, get names
        c_count = self.recording[self.signal].shape[0]
        if self.recording[self.signal].chans is None:
            channel_names = [''] * c_count
        else:
            channel_names=self.recording[self.signal].chans[:c_count]
        skip_channels = ['baseline']
        if channel_names is not None:
            keep = np.array([(n not in skip_channels) for n in channel_names])
            channel_names = [channel_names[i] for i in range(c_count) if keep[i]]
        else:
            keep = np.ones(c_count, dtype=bool)
            channel_names = None
        self.keep = keep
        self.channel_names = channel_names

        p = self.parent

        d = sig_array[self.keep, :]

        if self.point:
            self.axes.imshow(d, aspect='auto', cmap='Greys',
                             interpolation='nearest', origin='lower')
            self.axes.get_yaxis().set_visible(False)
        elif self.tiled:
            self.axes.imshow(d, aspect='auto', origin='lower')
        else:
            self.axes.plot(d.T)
            if self.channel_names is not None:
                if len(self.channel_names) > 1:
                    self.axes.legend(self.channel_names, frameon=False)
            self.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)

        self.axes.set_xlim(p.start_time*self.fs, p.stop_time*self.fs)
        self.axes.set_ylabel(self.signal)
        ax_remove_box(self.axes)
        self.draw()

        tick_labels = self.axes.get_xticklabels()
        if self.point or self.tiled:
            new_labels = ['']*len(tick_labels)
            self.axes.set_xticklabels(new_labels)
            self.draw()
        else:
            # TODO: Still not working... Should turn bins to seconds
            fmt = tkr.FuncFormatter(self.seconds_formatter())
            self.axes.yaxis.set_major_formatter(fmt)
            self.draw()

    def compute_initial_figure(self):
        pass

    def seconds_formatter(self):
        def fmt(x, pos):
            s = '{}'.format(x / self.fs)
            return s
        return fmt

    def update_figure(self):
        p = self.parent
        self.axes.set_xlim(p.start_time*self.fs, p.stop_time*self.fs)
        if not (self.point or self.tiled):
            self.axes.set_ylim(ymin=self.ymin, ymax=self.ymax)
        self.draw()


class EpochCanvas(MyMplCanvas):

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.recording = recording
        self.signal = signal
        self.parent = parent
        print("creating epoch canvas: {}".format(signal))
        self.max_time = 0
        self.epoch_groups = {}

    def update_figure(self):
        self.axes.cla()

        epochs = self.recording.epochs
        p = self.parent
        valid_epochs = epochs[(epochs['start'] >= p.start_time) &
                              (epochs['end'] < p.stop_time)]
        if valid_epochs.size == 0:
            print('no valid epochs')
            # valid_epochs = valid_epochs.append([{'name': 'EXPT', 'start': p.start_time, 'end': p.stop_time}])
            return

        # On each refresh, keep the same keys but reform the lists of indices.
        self.epoch_groups = {k: [] for k in self.epoch_groups}
        for i, r in valid_epochs.iterrows():
            s = r['start']
            e = r['end']
            n = r['name']

            prefix = n.split('_')[0]
            if prefix in ['PreStimSilence', 'PostStimSilence',
                          'REFERENCE','TARGET']:
                # skip
                pass
            elif prefix in self.epoch_groups:
                self.epoch_groups[prefix].append(i)
            else:
                self.epoch_groups[prefix] = [i]

        colors = ['Red', 'Orange', 'Green', 'LightBlue',
                  'DarkBlue', 'Purple', 'Pink', 'Black', 'Gray']
        i = 0
        for i, g in enumerate(self.epoch_groups):
            for j in self.epoch_groups[g]:
                n = valid_epochs['name'][j]
                s = valid_epochs['start'][j]
                e = valid_epochs['end'][j]

                try:
                    n2 = valid_epochs['name'][j+1]
                    s2 = valid_epochs['start'][j+1]
                    e2 = valid_epochs['end'][j+1]
                except KeyError:
                    # j is already the last epoch in the list
                    pass
                    n2 = n
                    s2 = s
                    e2 = e

                # If two epochs with the same name overlap,
                # extend the end of the first to the end of the second
                # and skip the second epoch.
                # Same if end goes past next start.
                if n == n2:
                    if (s2 < e) or (e > s2):
                        e = e2
                        j += 1
                    else:
                        pass

                # Don't plot text boxes outside of plot limits
                if s < p.start_time:
                    s = p.start_time
                elif e > p.stop_time:
                    e = p.stop_time

                x = np.array([s, e])
                y = np.array([i, i])

                self.axes.plot(x, y, '-', color=colors[i % len(colors)])
                self.axes.text(s, i, n, va='bottom', fontsize='small',
                               color=colors[i % len(colors)])

        self.axes.set_xlim([p.start_time, p.stop_time])
        self.axes.set_ylim([-0.5, i+0.5])
        ax_remove_box(self.axes)
        self.draw()

        xtick_labels = self.axes.get_xticklabels()
        ytick_labels = self.axes.get_yticklabels()
        new_xlabels = ['']*len(xtick_labels)
        new_ylabels = ['']*len(ytick_labels)
        self.axes.set_xticklabels(new_xlabels)
        self.axes.set_yticklabels(new_ylabels)
        self.axes.set_ylabel('epochs')
        self.draw()
