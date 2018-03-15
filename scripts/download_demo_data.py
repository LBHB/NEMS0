import logging
import os
import requests
from nems.recording import Recording

log = logging.getLogger(__name__)

# NOTE to LBHB team:
#    If additional sample data is added in the future, just
#    add the file names here -- *not* including any s3 prefixes
#    (i.e. nemspublic/sample_data/ferret1.tar.gz should just
#     be ferret1.tar.gz)
#    Additionally, any sample data specified here should be in
#    a zipped format, *not* a .mat, .csv etc.
DEMO_NAMES = [
        'TAR010c-18-1.tar.gz', 'eno052d-a1.tgz'
        ]


def get_demo_recordings(directory=None, unpack=False):
    '''
    Saves all sample recordings in the LBHB public s3 bucket to
    nems/recordings/, or to the specified directory. By default,
    the recordings will be kept in a compressed format; however,
    specifying unpack=True will instead save them uncompressed
    in a subdirectory.
    '''
    if directory is None:
        directory = os.path.abspath('../recordings')
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o666)
    names = DEMO_NAMES
    prefix = 'https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/'
    uris = [(prefix + n) for n in names]
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
            # TODO: clean this up, copied from recordings code.
            #       All of these content-types have showed up *so far*
            if not (r.status_code == 200 and
                    (r.headers['content-type'] == 'application/gzip' or
                     r.headers['content-type'] == 'application/x-gzip' or
                     r.headers['content-type'] == 'application/x-compressed' or
                     r.headers['content-type'] == 'application/x-tar' or
                     r.headers['content-type'] == 'application/x-tgz')):
                log.info('got response: {}, {}'
                         .format(r.headers, r.status_code))
                raise Exception('Error loading from uri: {}'.format(uri))
            with open(local, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)


# TODO: make this smarter and use argparse? or remove command line
#       arguments entirely and convert to a utility function for
#       other demo scripts?
def main():
    directory = input("Specify a directory (ENTER for default: ../recordings)")
    if not directory:
        directory = None
    unpack = input("Unpack the data? (TRUE to uncompress, FALSE otherwise)")
    get_demo_recordings(directory, unpack)


if __name__ == '__main__':
    main()
