import os
import nems.uri
from nems.recording import get_demo_recordings

# NOTE to LBHB team:
#    If additional sample data is added in the future, just
#    add the file names here -- *not* including any s3 prefixes
#    (i.e. nemspublic/sample_data/ferret1.tar.gz should just
#     be ferret1.tar.gz)
#    Additionally, any sample data specified here should be in
#    a zipped format, *not* a .mat, .csv etc.


def main():
    import argparse

    relative_signals_dir = '/../recordings'
    signals_dir = os.path.dirname(nems.uri.__file__) + relative_signals_dir
    signals_dir = os.path.abspath(signals_dir)

    parser = argparse.ArgumentParser('Download NEMS demo data')
    parser.add_argument('--directory', type=str, default=signals_dir,
                        help='Storage location for recordings (defaults to <nemspath>/recordings)',
                        )
    parser.add_argument(
        '--unpack', action='store_true',
        help='Recordings compressed by default, set --unpack to decompress'
        )
    args = parser.parse_args()
    directory = args.directory if args.directory else None
    unpack = args.unpack if args.unpack else None
    get_demo_recordings(directory, unpack)


if __name__ == '__main__':
    main()
