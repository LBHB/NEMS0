import io
import os
import gzip
import time
import tarfile
import logging
import requests
import pandas as pd
import numpy as np
import copy
import tempfile
import shutil

from nems.uri import local_uri, http_uri, targz_uri
import nems.epoch as ep
from nems.signal import SignalBase, RasterizedSignal, merge_selections, \
                        list_signals, load_signal, load_signal_from_streams

log = logging.getLogger(__name__)


class Recording:

    def __init__(self, signals):
        '''
        Signals argument should be a dictionary of signal objects.
        '''
        self.signals = signals

        # Verify that all signals are from the same recording
        recordings = [s.recording for s in self.signals.values()]
        if not recordings:
            raise ValueError('A recording must contain at least 1 signal')
        if not len(set(recordings)) == 1:
            raise ValueError('Not all signals are from the same recording.')

        self.name = recordings[0]
        self.uri = None  # This will be lost on copying

    def copy(self):
        '''
        Returns a copy of this recording.
        '''
        signals = self.signals.copy()
        other = Recording(signals)
        for k, v in vars(self).items():
            if k == 'signals':
                continue
            setattr(other, k, copy.copy(v))
        return other

    @property
    def epochs(self):
        '''
        The epochs of a recording is the superset of all signal epochs.
        '''
        # Merge the epochs. Be sure to ignore index since it's just a standard
        # sequential index for each signal's epoch (e.g., index 1 in signal1 has
        # no special meaning compared to index 1 in signal2). Drop all
        # duplicates since we sometimes replicate epochs across signals and
        # return the sorted values.
        epoch_set = [s.epochs for s in self.signals.values()]
        df = pd.concat(epoch_set, ignore_index=True)
        df.drop_duplicates(inplace=True)
        df.sort_values('start', inplace=True)
        df.index = np.arange(len(df))
        return df

    # Defining __getitem__ and __setitem__ make recording objects behave
    # like dictionaries when subscripted. e.g. recording['signal_name']
    # instead of recording.get_signal('signal_name').
    # See: https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types

    def __getitem__(self, key):
        return self.get_signal(key)

    def __setitem__(self, key, val):
        val.name = key
        self.add_signal(val)

    @staticmethod
    def load(uri):
        '''
        Loads from a local .tar.gz file, a local directory, from s3,
        or from an HTTP URL containing a .tar.gz file. Examples:

        # Load all signals in the gus016c-a2 directory
        rec = Recording.load('/home/myuser/gus016c-a2')
        rec = Recording.load('file:///home/myuser/gus016c-a2')

        # Load the local tar gz directory.
        rec = Recording.load('file:///home/myuser/gus016c-a2.tar.gz')

        # Load a tar.gz file served from a flat filesystem
        rec = Recording.load('http://potoroo/recordings/gus016c-a2.tar.gz')

        # Load a tar.gz file created by the nems-baphy interafce
        rec = Recording.load('http://potoroo/baphy/271/gus016c-a2')

        # Load from S3:
        rec = Recording.load('s3://nems.lbhb... TODO')
        '''
        if local_uri(uri):
            if targz_uri(uri):
                rec = Recording.load_targz(local_uri(uri))
            else:
                rec = Recording.load_dir(local_uri(uri))
        elif http_uri(uri):
            rec = Recording.load_url(http_uri(uri))
        elif uri[0:6] == 's3://':
            raise NotImplementedError
        else:
            raise ValueError('Invalid URI: {}'.format(uri))
        rec.uri = uri
        return rec

    @staticmethod
    def load_dir(directory_or_targz):
        '''
        Loads all the signals (CSV/JSON pairs) found in DIRECTORY or
        .tar.gz file, and returns a Recording object containing all of them.
        '''
        if os.path.isdir(directory_or_targz):
            files = list_signals(directory_or_targz)
            basepaths = [os.path.join(directory_or_targz, f) for f in files]
            signals = [load_signal(f) for f in basepaths]
            signals_dict = {s.name: s for s in signals}
            return Recording(signals=signals_dict)
        else:
            m = 'Not a directory: {}'.format(directory_or_targz)
            raise ValueError(m)

    @staticmethod
    def load_targz(targz):
        if os.path.exists(targz):
            with open(targz, 'rb') as stream:
                return load_recording_from_targz_stream(stream)
        else:
            m = 'Not a .tar.gz file: {}'.format(targz)
            raise ValueError(m)

    @staticmethod
    def load_url(url):
        '''
        Loads the recording object from a URL. File must be tar.gz format.
        '''
        r = requests.get(url, stream=True)
        if not (r.status_code == 200 and
                (r.headers['content-type'] == 'application/gzip' or
                 r.headers['content-type'] == 'text/plain' or
                 r.headers['content-type'] == 'application/x-gzip' or
                 r.headers['content-type'] == 'application/x-compressed' or
                 r.headers['content-type'] == 'application/x-tar' or
                 r.headers['content-type'] == 'application/x-tgz')):
            log.info('got response: %s, %d', r.headers, r.status_code)
            m = 'Error loading URL: {}'.format(url)
            log.error(m)
            raise Exception(m)
        obj = io.BytesIO(r.raw.read()) # Not sure why I need this!
        return load_recording_from_targz_stream(obj)

    @staticmethod
    def load_from_arrays(arrays, rec_name, fs, sig_names=None,
                         signal_kwargs={}):
        '''
        Generates a recording object, and the signal objects it contains,
        from a list of array-like structures of the form channels x time
        (see signal.py for more details about how arrays are represented
         by signals).

        If any of the arrays are more than 2-dimensional,
        an error will be thrown. Also pay close attention to any
        RuntimeWarnings from the signal class regarding improperly-shaped
        arrays, which may indicate that an array was passed as
        time x channels instead of the reverse.

        Arguments:
        ----------
        arrays : list of array-like
            The data to be converted to a recording of signal objects.
            Each item should be 2-dimensional and convertible to a
            numpy ndarray via np.array(x). No constraints are enforced
            on the dtype of the arrays, but in general float values
            are expected by most native NEMS functions.

        rec_name : str
            The name to be given to the new recording object. This will
            also be assigned as the recording attribute of each new signal.

        fs : int or list of ints
            The frequency of sampling of the data arrays - used to
            interconvert between real time and time bins (see signal.py).
            If int, the same fs will be assigned to each signal.
            If list, the length must match the length of arrays.

        sig_names : list of strings (optional)
            Name to attach to the signal created from
            each array. The length of this list should match that of
            arrays.
            If not specified, the signals will be given the generic
            names: ['signal1', 'signal2', ...].

        signal_kwargs : list of dicts
            Keyword arguments to be passed through to
            each signal object. The length of this list should
            match the length of arrays, and may be padded with empty
            dictionaries to ensure this constraint.
            For example:
                [{'chans': ['1 kHz', '3 kHz']}, {'chans': ['one', 'two']}, {}]
            Would assign channel names '1 kHz' and '3 kHz' to the signal
            for the first array, 'one' and 'two' for the second array,
            and no channel names (or any other arguments) for the third array.

            Valid keyword arguments are: chans, epochs, meta,
                                         and safety_checks

        Returns:
        --------
        rec : recording object
            New recording containing a signal for each array.
        '''
        # Assemble and validate lists for signal construction
        arrays = [np.array(a) for a in arrays]
        for i, a in enumerate(arrays):
            if len(a.shape) != 2:
                raise ValueError("Arrays should have shape chans x time."
                                 "Array {} had shape: {}"
                                 .format(i, a.shape))
        n = len(arrays)
        recs = [rec_name]*len(arrays)
        if sig_names:
            if not len(sig_names) == n:
                raise ValueError("Length of sig_names must match"
                                 "the length of arrays.\n"
                                 "Got sig_names: {} and arrays: {}"
                                 .format(len(sig_names), n))
        else:
            sig_names = ['sig%s'%i for i in range(n)]
        if isinstance(fs, int):
            fs = [fs]*n
        else:
            if not len(fs) == n:
                raise ValueError("Length of fs must match"
                                 "the length of arrays.\n"
                                 "Got fs: {} and arrays: {}"
                                 .format(len(fs), n))
        if not signal_kwargs:
            signal_kwargs = [{}]*n
        else:
            if not len(signal_kwargs) == n:
                raise ValueError("Length of signal_kwargs must match"
                                 "the length of arrays.\n"
                                 "Got signal_kwargs: {} and arrays: {}"
                                 .format(len(signal_kwargs), n))

        # Construct the signals
        to_sigs = zip(fs, arrays, sig_names, recs, signal_kwargs)
        signals = [
                RasterizedSignal(fs, a, name, rec, **kw)
                for fs, a, name, rec, kw in to_sigs
                ]
        signals = {s.name:s for s in signals}
        # Combine into recording and return
        return Recording(signals)

    def save(self, uri, uncompressed=False):
        '''
        Saves this recording to a URI as a compressed .tar.gz file.
        Returns the URI of what was saved, or None if there was a problem.

        Optional argument 'uncompressed' may be used to force the save
        to occur as a directory full of uncompressed files, but this only
        works for URIs that point to the local filesystem.

        For example:

        # Save to a local directory, use automatic filename
        rec.save('/home/username/recordings/')

        # Save it to a local file, with a specific name
        rec.save('/home/username/recordings/my_recording.tar.gz')

        # Same, but with an explicit file:// prefix
        rec.save('file:///home/username/recordings/my_recording.tar.gz')

        # Save it to the nems_db running on potoroo, use automatic filename
        rec.save('http://potoroo/recordings/')

        # Save it to the nems_db running on potoroo, specific filename
        rec.save('http://potoroo/recordings/my_recording.tar.gz')

        # Save it to AWS (TODO, Not Implemented, Needs credentials)
        rec.save('s3://nems.amazonaws.com/somebucket/')
        '''
        guessed_filename = self.name + '.tar.gz'
        # Set the URI metadata since we are writing to a URI now
        if not self.uri:
            self.uri = uri
        if local_uri(uri):
            uri = local_uri(uri)
            print(uri)
            if targz_uri(uri):
                return self.save_targz(uri)
            elif uncompressed:
                return self.save_dir(uri)
            else:
                #print(uri + '/' + guessed_filename)
                return self.save_targz(uri + '/' + guessed_filename)
        elif http_uri(uri):
            uri = http_uri(uri)
            if targz_uri(uri):
                return self.save_url(uri)
            elif uri[-1] == '/':
                return self.save_url(uri + guessed_filename)
            else:
                return self.save_url(uri + '/' + guessed_filename)
        elif uri[0:6] == 's3://':
            raise NotImplementedError
        else:
            raise ValueError('Invalid URI: {}'.format(uri))

    def save_dir(self, directory):
        '''
        Saves all the signals (CSV/JSON pairs) in this recording into
        DIRECTORY in a new directory named the same as this recording.
        '''
        if os.path.isdir(directory):
            directory = os.path.join(directory, self.name)
        elif os.path.exits(directory):
            m = 'File named {} exists; unable to create dir'.format(directory)
            raise ValueError(m)
        else:
            os.mkdir(directory)
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        for s in self.signals.values():
            s.save(directory)
        return directory

    def save_targz(self, uri):
        '''
        Saves all the signals (CSV/JSON pairs) in this recording
        as a .tar.gz file at a local URI.
        '''
        directory = os.path.dirname(uri)
        if not os.path.isdir(directory):
            os.makedirs(directory, mode=0o0777)
        os.umask(0o0000)
        with open(uri, 'wb') as archive:
            tgz = self.as_targz()
            archive.write(tgz.read())
            tgz.close()
        return uri

    def as_targz(self):
        '''
        Returns a BytesIO containing all the rec's signals as a .tar.gz stream.
        You may either send this over HTTP or save it to a file. No temporary
        files are created in the creation of this .tar.gz stream.

        Example of saving an in-memory recording to disk:
            rec = Recording(...)
            with open('/some/path/test.tar.gz', 'wb') as fh:
                tgz = rec.as_targz()
                fh.write(tgz.read())
                tgz.close()  # Don't forget to close it!
        '''
        f = io.BytesIO()  # Create a buffer
        tar = tarfile.open(fileobj=f, mode='w:gz')
        # tar = tarfile.open('/home/ivar/poopy.tar.gz', mode='w:gz')
        # With the tar buffer open, write all signal files
        for s in self.signals.values():
            d = s.as_file_streams()  # Dict mapping filenames to streams
            for filename, stringstream in d.items():
                if type(stringstream) is io.BytesIO:
                    stream = stringstream
                else:
                    stream = io.BytesIO(stringstream.getvalue().encode())
                info = tarfile.TarInfo(os.path.join(self.name, filename))
                info.uname = 'nems'  # User name
                info.gname = 'users'  # Group name
                info.mtime = time.time()
                info.size = stream.getbuffer().nbytes
                tar.addfile(info, stream)
        tar.close()
        f.seek(0)
        return f

    def save_url(self, uri, compressed=False):
        '''
        Saves this recording to a URL. Returns the URI if it succeeded,
        else None. Check the return code to see if the URL save failed
        and you need to save locally instead. e.g.

        # Example: Try to save remotely, or save locally if it fails
        if not rec.save_url(url):
             rec.save('/tmp/')   # Save to /tmp as a fallback
        '''
        r = requests.put(uri, data=self.as_targz())
        if r.status_code == 200:
            return uri
        else:
            m = 'HTTP PUT failed (Code: {}) for {}.'.format(r.status_code,
                                                            uri)
            log.warn(m)
            return None

    def get_signal(self, signal_name):
        '''
        Returns the signal object with the given signal_name, or None
        if it was was found.

        signal_name should be a string
        '''
        if signal_name in self.signals:
            return self.signals[signal_name]
        else:
            return None

    def add_signal(self, signal):
        '''
        Adds the signal equal to this recording. Any existing signal
        with the same name will be overwritten. No return value.
        '''
        if not isinstance(signal, SignalBase):
            raise TypeError("Recording signals must be instances of"
                            " a Signal class. signal {} was type: {}"
                            .format(signal.name, type(signal)))
        self.signals[signal.name] = signal

    def _split_helper(self, fn):
        '''
        For internal use only by the split_* functions.
        '''
        est = {}
        val = {}
        for s in self.signals.values():
            (e, v) = fn(s)
            est[e.name] = e
            val[v.name] = v
        return (Recording(signals=est), Recording(signals=val))

    def split_at_time(self, fraction):
        '''
        Calls .split_at_time() on all signal objects in this recording.
        For example, fraction = 0.8 will result in two recordings,
        with 80% of the data in the left, and 20% of the data in
        the right signal. Useful for making est/val data splits, or
        truncating the beginning or end of a data set.
        '''
        return self._split_helper(lambda s: s.split_at_time(fraction))

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        '''
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. Both est and val will have non-matching data NaN'd out.
        '''
        return self._split_helper(lambda s: s.split_by_epochs(epochs_for_est,
                                                              epochs_for_val))

    def split_using_epoch_occurrence_counts(self, epoch_regex):
        '''
        Returns (est, val) given a recording rec, a signal name 'stim_name', and an
        epoch_regex that matches 'various' epochs. This function will throw an exception
        when there are not exactly two values for the number of epoch occurrences; i.e.
        low-rep epochs and high-rep epochs.

        NOTE: This is a fairly specialized function that we use in the LBHB lab. We have
        found that, given a limited recording time, it is advantageous to have a variety of sounds
        presented to the neuron (i.e. many low-repetition stimuli) for accurate estimation
        of its parameters. However, during the validation process, it helps to have many
        repetitions of the same stimuli so that we can more accurately estimate the peri-
        stimulus time histogram (PSTH). This function tries to split the data into those
        two data sets based on the epoch occurrence counts.
        '''
        groups = ep.group_epochs_by_occurrence_counts(self.epochs, epoch_regex)
        if len(groups) > 2:
            l=np.array(list(groups.keys()))
            k=l>np.mean(l)
            hi=np.max(l[k])
            lo=np.min(l[k==False])

            # generate two groups
            g = {hi: [], lo: []}
            for i in list(np.where(k)[0]):
                g[hi] = g[hi] + groups[l[i]]
            for i in list(np.where(k == False)[0]):
                g[lo] = g[lo] + groups[l[i]]
            groups = g

        elif len(groups)==1:
            k = list(groups.keys())[0]
            g1 = groups[k]
            n = len(g1)
            vset = np.int(np.floor(n*0.8))

            g={1: g1[:vset], 2: g1[vset:]}
            groups = g

        elif len(groups) == 0:
            m = "No occurrences?? Unable to split recording into est/val sets"
            m += str(groups)
            raise ValueError(m)

        n_occurrences = sorted(groups.keys())
        lo_rep_epochs = groups[n_occurrences[0]]
        hi_rep_epochs = groups[n_occurrences[1]]
        return self.split_by_epochs(lo_rep_epochs, hi_rep_epochs)

    def get_epoch_indices(self, epoch_name):

        keys = list(self.signals.keys())
        if 'mask' not in keys:
            epochs = self[keys[0]].get_epoch_indices(epoch_name)

        else:
            # only keep epoch matching mask
            m_data = self['mask'].as_continuous().copy()
            all_epochs = self['mask'].get_epoch_indices(epoch_name)

            epochs = np.zeros([0, 2], dtype=np.int32)
            for lb, ub in all_epochs:
                if np.sum(1 - (m_data[0, lb:ub])) == 0:
                    epochs = np.append(epochs, [[lb, ub]], axis=0)

        return epochs

    def jackknife_mask_by_epoch(self, njacks, jack_idx, epoch_name,
                                tiled=True, invert=False):
        '''
        Creates mask or updates existing mask, with subset of epochs
          matching epoch_name set to False

        Optional argument 'invert' causes everything BUT the matched epochs
        to be NaN'd. njacks determines the number of jackknifes to divide
        the epochs into, and jack_idx determines which one to return.

        'Tiled' makes each jackknife use every njacks'th occurrence, and is
        probably best explained by the following example...

        If there are 18 occurrences of an epoch, njacks=5, invert=False,
        and tiled=True, then the five jackknifes will have these
        epochs NaN'd out:

           jacknife[0]:  0, 5, 10, 15
           jacknife[1]:  1, 6, 11, 16
           jacknife[2]:  2, 7, 12, 17
           jacknife[3]:  3, 8, 13
           jacknife[4]:  4, 9, 14

        Note that the last two jackknifes have one fewer occurrences.

        If tiled=False, then the pattern of NaN'd epochs becomes sequential:

           jacknife[0]:   0,  1,  2,  3
           jacknife[1]:   4,  5,  6,  7,
           jacknife[2]:   8,  9, 10, 11,
           jacknife[3]:  12, 13, 14, 15,
           jacknife[4]:  16, 17

        Here we can see the last jackknife has 2 fewer occurrences.
        '''

        # create mask if one doesn't exist yet
        if 'mask' not in self.signals.keys():
            rec = self.create_mask(True)
        else:
            rec = self.copy()

        # initialize mask to be all True
        m_data = rec['mask'].as_continuous().copy()

        # find all matching epochs
        epochs = self.get_epoch_indices(epoch_name)
        occurrences = epochs.shape[0]

        if occurrences == 0:
            m = 'No epochs found matching epoch_name. Unable to jackknife.'
            raise ValueError(m)

        if occurrences < njacks:
            raise ValueError("Can't divide {0} occurrences into {1} jackknifes"
                             .format(occurrences, njacks))

        if jack_idx < 0 or njacks < 0:
            raise ValueError("Neither jack_idx nor njacks may be negative")

        nrows = np.int(np.ceil(occurrences / njacks))
        idx_data = np.arange(nrows * njacks)

        if tiled:
            idx_data = idx_data.reshape(nrows, njacks)
            idx_data = np.swapaxes(idx_data, 0, 1)
        else:
            idx_data = idx_data.reshape(njacks, nrows)

        # jmask = bins that should be excluded, on top of whatever is already
        # False in m_data
        jmask = np.zeros_like(m_data, dtype=np.bool)
        for idx in idx_data[jack_idx].tolist():
            if idx < occurrences:
                lb, ub = epochs[idx]
                jmask[:, lb:ub] = True

        if invert:
            jmask = ~jmask

        m_data[jmask] = False

        rec['mask'] = rec['mask']._modified_copy(m_data)

        return rec

    def jackknife_by_epoch(self, njacks, jack_idx, epoch_name,
                           tiled=True,invert=False,
                           only_signals=None, excise=False):
        '''
        By default, calls jackknifed_by_epochs on all signals and returns a new
        set of data. If you would only like to jackknife certain signals,
        while copying all other signals intact, provide their names in a
        list to optional argument 'only_signals'.
        '''
        if excise and only_signals:
            raise Exception('Excising only some signals makes signals ragged!')
        new_sigs = {}
        for sn in self.signals.keys():
            if (not only_signals or sn in set(only_signals)):
                s = self.signals[sn]
                log.debug("JK: {0} {1}/{2} {3}".format(s.name,jack_idx,
                          njacks,epoch_name))
                new_sigs[sn] = s.jackknife_by_epoch(njacks, jack_idx,
                                                    epoch_name=epoch_name,
                                                    invert=invert, tiled=tiled)
        return Recording(signals=new_sigs)

        # if signal_names is not None:
        #     signals = {n: self.signals[n] for n in signal_names}
        # else:
        #     signals = self.signals

        # kw = dict(regex=regex, invert=invert)
        # split = {n: s.jackknifed_by_epochs(**kw) for n, s in signals.items()}
        # return Recording(signals=split)

    def jackknife_by_time(self, nsplits, split_idx, only_signals=None,
                          invert=False, excise=False):
        '''
        By default, calls jackknifed_by_time on all signals and returns a new
        set of data.  If you would only like to jackknife certain signals,
        while copying all other signals intact, provide their names in a
        list to optional argument 'only_signals'.
        '''
        if excise and only_signals:
            raise Exception('Excising only some signals makes signals ragged!')
        new_sigs = {}
        for sn in self.signals.keys():
            if (not only_signals or sn in set(only_signals)):
                s = self.signals[sn]
                new_sigs[sn] = s.jackknife_by_time(nsplits, split_idx,
                                                   invert=invert, excise=excise)
        return Recording(signals=new_sigs)

# moved to independent function, below
#    @staticmethod
#    def jackknife_inverse_merge(rec_list):
#        '''
#        merges list of jackknife validation data into a signal recording
#        '''
#        if type(rec_list) is not list:
#            raise ValueError('Expecting list of recordings')
#        new_sigs = {}
#        rec1=rec_list[0]
#        for sn in rec1.signals.keys():
#            sig_list=[r[sn] for r in rec_list]
#            #new_sigs[sn]=sig_list[0].jackknife_inverse_merge(sig_list)
#            new_sigs[sn]=merge_selections(sig_list)
#        return Recording(signals=new_sigs)

    def jackknifes_by_epoch(self, nsplits, epoch_name, only_signals=None):
        raise NotImplementedError         # TODO

    def jackknifes_by_time(self, nsplits, only_signals=None):
        raise NotImplementedError         # TODO

    def concatenate_recordings(self, recordings):
        '''
        Concatenate more recordings on to the end of this Recording,
        and return the result. Recordings must have identical signal
        names, channels, and fs, or an exception will be thrown.
        '''
        signal_names = self.signals.keys()
        for recording in recordings:
            if signal_names != recording.signals.keys():
                raise ValueError('Recordings do not contain same signals')

        # Merge the signals and return it as a new recording.
        merged_signals = {}
        for signal_name in signal_names:
            signals = [r.signals[signal_name] for r in recordings]
            merged_signals[signal_name] = Signal.concatenate_time(signals)

        # TODO: copy the epochs as well
        raise NotImplementedError    # TODO

        return Recording(merged_signals)

        # TODO: copy the epochs as well
    def select_epoch():
        raise NotImplementedError    # TODO

    def select_times(self, times, padding=0):

        if padding != 0:
            raise NotImplementedError    # TODO

        k = list(self.signals.keys())
        newsigs = {n: s.select_times(times) for n, s in self.signals.items()}

        return Recording(newsigs)

    def nan_times(self, times, padding=0):

        if padding != 0:
            raise NotImplementedError    # TODO

        k = list(self.signals.keys())
        newsigs = {n: s.nan_times(times) for n, s in self.signals.items()}

        return Recording(newsigs)

    def create_mask(self, epoch=None, base_signal=None):
        '''
        Initialize mask signal to False for all times

        TODO: remove unnecessary deepcopys from this and subsequent functions
        TODO: add epochs, base signal parameters
           if epochs not None, call self.or_mask
        '''

        rec = self.copy()
        if base_signal is None:
            sig_name = list(rec.signals.keys())[0]
            base_signal = rec[sig_name]

        if epoch is None:
            mask = np.zeros([1, base_signal.ntimes], dtype=np.bool)
        else:
            mask = base_signal.generate_epoch_mask(epoch)

        try:
            mask_sig = base_signal._modified_copy(mask)
        except AttributeError:
            # Only rasterized signals support _modified_copy
            mask_sig = base_signal.rasterize()._modified_copy(mask)
        mask_sig.name = 'mask'

        rec.add_signal(mask_sig)

        return rec

    def or_mask(self, epoch, invert=False):
        '''
        epoch : string, list of strings, 2darray
           epoch(s) can be either list of epoch names or simply an array
        of index segments to mask

        Make rec['mask'] == True for all epochs (or segs) in list of epoch
        names.

        ex:
            rec.or_mask(['HIT_TRIAL', 'PASSIVE_EXPERIMENT']) will return a
            new recording with rec['mask'] == True for all PASSIVE EXPERIMENT
            and all HIT TRIAL epochs
        '''
        if 'mask' not in self.signals.keys():
            rec = self.create_mask(False)
        else:
            rec = self.copy()

        if (type(epoch) is str) or (type(epoch) is np.ndarray):
            or_mask = rec['mask'].generate_epoch_mask(epoch)

        elif type(epoch) is list:
            or_mask = rec['mask'].generate_epoch_mask(epoch[0])

            for e in epoch[1:]:
                or_mask = or_mask | rec['mask'].generate_epoch_mask(e)

        # Invert
        if invert:
            or_mask = ~or_mask

        # apply or_mask to existing mask
        m = rec['mask'].as_continuous()
        rec['mask'] = rec['mask']._modified_copy(m | or_mask)

        return rec

    def and_mask(self, epoch, invert=False):
        '''
        list of epoch names can be either list of epoch names or simply an array
        of index segments to mask


        Make mask == True for all epochs (or index segments) in list of epoch
        names where current mask is also true.

        example use:
            newrec = rec.or_mask(['ACTIVE_EXPERIMENT'])
            newrec = rec.and_mask(['REFERENCE', 'TARGET'])

            newrec['mask'] == True only during REFERENCE and TARGET epochs
            contained within ACTIVE_EXPERIMENT epochs
        '''
        if 'mask' not in self.signals.keys():
            rec = self.create_mask(True)
        else:
            rec = self.copy()

        if (type(epoch) is str) or (type(epoch) is np.ndarray):
            and_mask = rec['mask'].generate_epoch_mask(epoch)

        elif type(epoch) is list:
            and_mask = rec['mask'].generate_epoch_mask(epoch[0])

            for e in epoch[1:]:
                and_mask = and_mask | rec['mask'].generate_epoch_mask(e)

        # Invert
        if invert:
            and_mask = ~and_mask

        # apply and_mask to existing mask
        m = rec['mask'].as_continuous()
        rec['mask'] = rec['mask']._modified_copy(m & and_mask)

        return rec

    def apply_mask(self):
        '''
        Used to excise data based on boolean called mask. Returns new recording
        with only data specified mask. To make mask, see "create_epoch_mask"
        '''
        if 'mask' not in self.signals.keys():
            raise ValueError('Need to create a mask signal first')

        rec = copy.deepcopy(self)
        sig = rec['mask']

        if np.sum(~sig._data) == 0:
            return rec

        s_indices = np.argwhere(np.diff(rec['mask']._data.squeeze())).squeeze()+1
        last_ind = len(s_indices)-1

        s = []
        e = []

        i = 0
        while i < last_ind:
            if i == 0:
                if rec['mask']._data.squeeze()[0] == True:
                    # print('hello')
                    s.append(0)
                    e.append(s_indices[i])
                    i+=1
                else:
                    s.append(s_indices[i])
                    e.append(s_indices[i+1])
                    i+=2
            else:
                s.append(s_indices[i])
                e.append(s_indices[i+1])
                i+=2

        times = (np.vstack((np.array(s), np.array(e)))/sig.fs).T
        if times[-1,1]==times[-1,0]:
            times = times[:-1,:]

        newrec = rec.select_times(np.array(np.array(times.tolist())))
        return newrec

## I/O functions
def load_recording_from_targz(targz):
    if os.path.exists(targz):
        with open(targz, 'rb') as stream:
            return load_recording_from_targz_stream(stream)
    else:
        m = 'Not a .tar.gz file: {}'.format(targz)
        raise ValueError(m)


def load_recording_from_targz_stream(tgz_stream):
    '''
    Loads the recording object from the given .tar.gz stream, which
    is expected to be a io.BytesIO object.
    For hdf5 files, copy to temporary directory and load with hdf5 utility
    '''
    tpath=None

    streams = {}  # For holding file streams as we unpack
    with tarfile.open(fileobj=tgz_stream, mode='r:gz') as t:
        for member in t.getmembers():
            if member.size == 0:  # Skip empty files
                continue
            basename = os.path.basename(member.name)
            # Now put it in a subdict so we can find it again
            signame = str(basename.split('.')[0:2])
            if basename.endswith('epoch.csv'):
                keyname = 'epoch_stream'
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))

            elif basename.endswith('.csv'):
                keyname = 'data_stream'
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))

            elif basename.endswith('.h5'):
                keyname = 'data_stream'
                #f_in = io.BytesIO(t.extractfile(member).read())

                # current non-optimal solution. extract hdf5 file to disk and then load
                if not tpath:
                    tpath=tempfile.mktemp()
                t.extract(member,tpath)
                f=tpath+'/'+member.name

            elif basename.endswith('.json'):
                keyname = 'json_stream'
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))

            else:
                m = 'Unexpected file found in tar.gz: {} (size={})'.format(member.name, member.size)
                raise ValueError(m)
            # Ensure that we can doubly nest the streams dict
            if signame not in streams:
                streams[signame] = {}
            # Read out a stringIO object for each file now while it's open
            #f = io.StringIO(t.extractfile(member).read().decode('utf-8'))
            streams[signame][keyname] = f

    # Now that the streams are organized, convert them into signals
    # log.debug({k: streams[k].keys() for k in streams})
    signals = [load_signal_from_streams(**sg) for sg in streams.values()]
    signals_dict = {s.name: s for s in signals}

    rec = Recording(signals=signals_dict)

    if tpath:
        shutil.rmtree(tpath) # clean up if tpath is not None

    return rec

def load_recording(uri):
    '''
    Loads from a local .tar.gz file, a local directory, from s3,
    or from an HTTP URL containing a .tar.gz file. Examples:

    # Load all signals in the gus016c-a2 directory
    rec = Recording.load('/home/myuser/gus016c-a2')
    rec = Recording.load('file:///home/myuser/gus016c-a2')

    # Load the local tar gz directory.
    rec = Recording.load('file:///home/myuser/gus016c-a2.tar.gz')

    # Load a tar.gz file served from a flat filesystem
    rec = Recording.load('http://potoroo/recordings/gus016c-a2.tar.gz')

    # Load a tar.gz file created by the nems-baphy interafce
    rec = Recording.load('http://potoroo/baphy/271/gus016c-a2')

    # Load from S3:
    rec = Recording.load('s3://nems.lbhb... TODO')
    '''
    if local_uri(uri):
        if targz_uri(uri):
            rec = load_recording_from_targz(local_uri(uri))
        else:
            rec = load_recording_from_dir(local_uri(uri))
    elif http_uri(uri):
        rec = load_recording_from_url(http_uri(uri))
    elif uri[0:6] == 's3://':
        raise NotImplementedError
    else:
        raise ValueError('Invalid URI: {}'.format(uri))
    rec.uri = uri

    # TODO ? create copy of 'stim' to 'pred' ?

    return rec

def load_recording_from_dir(directory_or_targz):
    '''
    Loads all the signals (CSV/JSON pairs) found in DIRECTORY or
    .tar.gz file, and returns a Recording object containing all of them.
    '''
    if os.path.isdir(directory_or_targz):
        files = list_signals(directory_or_targz)
        basepaths = [os.path.join(directory_or_targz, f) for f in files]
        signals = [load_signal(f) for f in basepaths]
        signals_dict = {s.name: s for s in signals}
        return Recording(signals=signals_dict)
    else:
        m = 'Not a directory: {}'.format(directory_or_targz)
        raise ValueError(m)

def load_recording_from_url(url):
    '''
    Loads the recording object from a URL. File must be tar.gz format.
    '''
    r = requests.get(url, stream=True)
    if not (r.status_code == 200 and
            (r.headers['content-type'] == 'application/gzip' or
             r.headers['content-type'] == 'text/plain' or
             r.headers['content-type'] == 'application/x-gzip' or
             r.headers['content-type'] == 'application/x-compressed' or
             r.headers['content-type'] == 'application/x-compressed-tar' or
             r.headers['content-type'] == 'application/x-tar' or
             r.headers['content-type'] == 'application/x-tgz')):
        log.info('got response: {}, {}'.format(r.headers, r.status_code))
        log.info('status_code: %d, content-type: %s', r.status_code, r.headers['content-type'])
        m = 'Error loading URL: {}'.format(url)
        log.error(m)
        raise Exception(m)
    obj = io.BytesIO(r.raw.read()) # Not sure why I need this!
    return load_recording_from_targz_stream(obj)

def load_recording_from_arrays(arrays, rec_name, fs, sig_names=None,
                     signal_kwargs={}):
    '''
    Generates a recording object, and the signal objects it contains,
    from a list of array-like structures of the form channels x time
    (see signal.py for more details about how arrays are represented
     by signals).

    If any of the arrays are more than 2-dimensional,
    an error will be thrown. Also pay close attention to any
    RuntimeWarnings from the signal class regarding improperly-shaped
    arrays, which may indicate that an array was passed as
    time x channels instead of the reverse.

    Arguments:
    ----------
    arrays : list of array-like
        The data to be converted to a recording of signal objects.
        Each item should be 2-dimensional and convertible to a
        numpy ndarray via np.array(x). No constraints are enforced
        on the dtype of the arrays, but in general float values
        are expected by most native NEMS functions.

    rec_name : str
        The name to be given to the new recording object. This will
        also be assigned as the recording attribute of each new signal.

    fs : int or list of ints
        The frequency of sampling of the data arrays - used to
        interconvert between real time and time bins (see signal.py).
        If int, the same fs will be assigned to each signal.
        If list, the length must match the length of arrays.

    sig_names : list of strings (optional)
        Name to attach to the signal created from
        each array. The length of this list should match that of
        arrays.
        If not specified, the signals will be given the generic
        names: ['signal1', 'signal2', ...].

    signal_kwargs : list of dicts
        Keyword arguments to be passed through to
        each signal object. The length of this list should
        match the length of arrays, and may be padded with empty
        dictionaries to ensure this constraint.
        For example:
            [{'chans': ['1 kHz', '3 kHz']}, {'chans': ['one', 'two']}, {}]
        Would assign channel names '1 kHz' and '3 kHz' to the signal
        for the first array, 'one' and 'two' for the second array,
        and no channel names (or any other arguments) for the third array.

        Valid keyword arguments are: chans, epochs, meta,
                                     and safety_checks

    Returns:
    --------
    rec : recording object
        New recording containing a signal for each array.
    '''
    # Assemble and validate lists for signal construction
    arrays = [np.array(a) for a in arrays]
    for i, a in enumerate(arrays):
        if len(a.shape) != 2:
            raise ValueError("Arrays should have shape chans x time."
                             "Array {} had shape: {}"
                             .format(i, a.shape))
    n = len(arrays)
    recs = [rec_name]*len(arrays)
    if sig_names:
        if not len(sig_names) == n:
            raise ValueError("Length of sig_names must match"
                             "the length of arrays.\n"
                             "Got sig_names: {} and arrays: {}"
                             .format(len(sig_names), n))
    else:
        sig_names = ['sig%s'%i for i in range(n)]
    if isinstance(fs, int):
        fs = [fs]*n
    else:
        if not len(fs) == n:
            raise ValueError("Length of fs must match"
                             "the length of arrays.\n"
                             "Got fs: {} and arrays: {}"
                             .format(len(fs), n))
    if not signal_kwargs:
        signal_kwargs = [{}]*n
    else:
        if not len(signal_kwargs) == n:
            raise ValueError("Length of signal_kwargs must match"
                             "the length of arrays.\n"
                             "Got signal_kwargs: {} and arrays: {}"
                             .format(len(signal_kwargs), n))

    # Construct the signals
    to_sigs = zip(fs, arrays, sig_names, recs, signal_kwargs)
    signals = [
            RasterizedSignal(fs, a, name, rec, **kw)
            for fs, a, name, rec, kw in to_sigs
            ]
    signals = {s.name:s for s in signals}
    # Combine into recording and return
    return Recording(signals)



## general methods

def jackknife_inverse_merge(rec_list):
    '''
    merges list of jackknife validation data into a signal recording

    currently two different approaches, depending on whether mask signal
    is present.
    '''
    if type(rec_list) is not list:
        raise ValueError('Expecting list of recordings')

    sig_list = list(rec_list[0].signals.keys())
    if 'mask' in sig_list:
        # new system: using mask==True to identify valid segment from
        # each signal  -- only pred and mask, since those are the only
        # ones that should be modified???
        new_sigs = {}

        #for sn in ['pred', 'mask', 'stim', 'psth']:
        for sn in sig_list:
            if sn in sig_list:
                r = rec_list[0][sn]
                if type(r._data) is np.ndarray:
                    _data = np.zeros(r.shape, dtype=r._data.dtype)
                    if not(_data.dtype == bool):
                        _data[:] = np.nan
                else:
                    _data = np.zeros(r.shape)
                    _data[:] = np.nan

                # print(sn)
                # print(np.sum(np.isfinite(_data)))
                for r in rec_list:
                    m = r['mask'].as_continuous()[0, :]
                    _data[:, m] = r[sn].rasterize().as_continuous()[:, m]
                    # if sn=='pred':
                    #    print(np.sum(m))
                    #    print(np.sum(np.isfinite(_data)))
                new_sigs[sn] = r[sn].rasterize()._modified_copy(_data)
                # print(np.sum(np.isfinite(new_sigs[sn].as_continuous())))
    else:
        new_sigs = {}
        for sn in sig_list:
            sig_list=[r[sn] for r in rec_list]
            #new_sigs[sn]=sig_list[0].jackknife_inverse_merge(sig_list)
            new_sigs[sn]=merge_selections(sig_list)

    return Recording(signals=new_sigs)


# TODO: Might be a better place for this, but moved it from nems.uri
#       for now because it was causing circular import issues since
#       the unpack option depends on code in this module.

DEMO_NAMES = [
        'TAR010c-18-1.tgz', 'eno052d-a1.tgz',
        'BRT026c-02-1.tgz', 'resp2.tgz',
        'TAR010c-06-1.tgz',
        ]

def get_demo_recordings(directory=None, name=None, unpack=False):
    '''
    Saves all sample recordings in the LBHB public s3 bucket to
    nems/recordings/, or to the specified directory. By default,
    the recordings will be kept in a compressed format; however,
    specifying unpack=True will instead save them uncompressed
    in a subdirectory.
    '''
    if name is None:
        names = DEMO_NAMES
    else:
        names = [name]

    prefix = 'https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/'
    uris = [(prefix + n) for n in names]

    if directory is None:
        nems_dir = os.path.abspath(os.path.dirname(__file__) + '/..')
        directory = nems_dir + '/recordings'

    if unpack:
        recs = [Recording.load(uri) for uri in uris]
        for rec in recs:
            log.info("Saving file at {} in {}".format(rec.uri, directory))
            rec.save_dir(directory)
    else:
        """
        https://stackoverflow.com/questions/16694907/
        how-to-download-large-file-in-python-with-requests-py
        """
        for uri in uris:
            file = uri.split('/')[-1]
            local = os.path.join(directory, file)
            if os.path.isfile(local):
                log.debug("Local file {} already exists, skipping."
                          .format(local))
            else:
                log.info("Saving file at {} to {}".format(uri, local))
                r = requests.get(uri, stream=True)
                # TODO: clean this up, copied from recordings code.
                #       All of these content-types have showed up *so far*
                allowed_headers = [
                        'application/gzip', 'application/x-gzip',
                        'application/x-compressed', 'application/x-tgz',
                        'application/x-tar', 'application/x-compressed-tar',
                        ]
                if not (r.status_code == 200
                        and r.headers['content-type'] in allowed_headers):
                    log.info('got response: {}, {}'
                             .format(r.headers, r.status_code))
                    raise Exception('Error loading from uri: {}'.format(uri))

                try:
                    with open(local, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                except PermissionError as e:
                    log.warn("Couldn't write in directory: \n{}\n"
                             "due to permission issues. Make sure the "
                             "parent directory grants write permission."
                             .format(directory))
                    log.exception(e)
    return directory