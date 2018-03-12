import logging
import os
import requests
from nems.recording import Recording

log = logging.getLogger(__name__)
DEMO_NAMES = [
        'TAR010c-18-1',
        ]

# TODO: getting 403 response when attempting to access nemspublic through
#       http request. What's causing this?
def get_demo_recordings(directory=None, unpack=False):
    if directory is None:
        directory = os.path.abspath('../recordings')
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o666)
    names = DEMO_NAMES
    prefix = 'https://nemspublic.s3.amazonaws.com/recordings/'
    uris = [(prefix + n + '.tar.gz') for n in names]
    if unpack:
        recs = [Recording.load(uri) for uri in uris]
        for rec in recs:
            rec.save_dir(directory)
    else:
        """
        https://stackoverflow.com/questions/16694907/
        how-to-download-large-file-in-python-with-requests-py
        """
        for uri in uris:
            file = uri.split('/')[-1]
            local = os.path.join(directory, file)
            log.info("Saving file at {} to {}".format(uri, local))
            r = requests.get(uri, stream=True)
            if not (r.status_code == 200 and
                    (r.headers['content-type'] == 'application/gzip' or
                     r.headers['content-type'] == 'application/x-tar' or
                     r.headers['content-type'] == 'application/x-tgz')):
                log.info('got response: {}, {}'.format(r.headers, r.status_code))
                m = 'Error loading URL: {}'.format(uri)
                log.error(m)
                raise Exception(m)
            with open(local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

if __name__ == '__main__':
    get_demo_recordings()