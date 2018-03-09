from nems.recording import Recording


def test_recording_loading():
    '''
    Test the loading and saving of files to various HTTP/S3/File routes.
    '''
    # NOTE: FOR THIS TEST TO SUCCEED, /auto/data/tmp/recordings/ must not have
    # blah.tar.gz or TAR010c-18-1.tar.gz in it.

    # Local filesystem
    rec0 = Recording.load("/home/ivar/git/nems/signals/TAR010c-18-1.tar.gz")
    rec1 = Recording.load("/auto/data/tmp/recordings/TAR010c-18-1.tar.gz")
    rec2 = Recording.load("file:///auto/data/tmp/recordings/TAR010c-18-1.tar.gz")

    # HTTP
    rec3 = Recording.load("http://potoroo:3001/recordings/TAR010c-18-1.tar.gz")
    rec4 = Recording.load("http://potoroo/baphy/271/TAR010c-18-1")

    # S3
    # Direct access (would need AWS CLI lib? Maybe not best idea!)
    # TODO: Requires s3 credentials in environment. Probably best if on server only; put in nems_db?
    rec5 = Recording.load('s3://mybucket/myfile.tar.gz')

    # Indirect access via http:
    rec6 = Recording.load('https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/TAR010c-18-1.tar.gz')

    # Save to a specific tar.gz file
    rec0.save('/home/ivar/tmp/tmp.tar.gz')

    # Save in a newly created directory under /home/ivar/tmp
    rec0.save('/home/ivar/tmp/', uncompressed=True)

    # Save ina  newly created tar.gz under /home/ivar/tmp
    rec0.save('/home/ivar/tmp/')

    if not rec0.save('http://potoroo/recordings/blah.tar.gz'):
        print('Error saving to explicit file URI')
        assert 0

    if not rec0.save('http://potoroo/recordings/'):
        print('Error saving to a directory URI')
        assert 0
