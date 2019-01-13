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
        self.axes.get_yaxis().set_visible(False)
        self.axes.get_xaxis().set_visible(False)
        #self.axes.set_position([0.175, 0.175, 0.775, 0.7])

        super(FigureCanvas, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, qw.QSizePolicy.Expanding,
                                   qw.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setContentsMargins(0, 0, 0, 0)
        #self.figure.tight_layout()
        self.figure.subplots_adjust(left=0, bottom=0, right=1,
                                    top=1, wspace=0, hspace=0)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)

class EpochCanvas(MyMplCanvas):

    def __init__(self, recording=None, signal='stim', parent=None,
                 *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.recording = recording
        self.signal = signal
        self.parent = parent
        self.max_time = 0
        self.epoch_groups = {}
        self.update_figure()

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
