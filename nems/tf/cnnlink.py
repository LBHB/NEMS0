
import nems
import nems.modelspec as ms
import nems.tf.cnn as cnn

def modelspec2cnn(modelspec):
    """convert NEMS modelspec to TF network.
    Initialize with existing phi?
    Translations:
        wc -> reweight, identity (require lvl?)
        fir+lvl -> conv, indentity
        wc+relu -> reweight, relu
        fir+relu -> conv2d, relu

    """
    pass

def cnn2modelspec(Net, modelspec=None):
    """pass TF network fit back into modelspec phi.
    Generate new modelspec if not provided"""

    pass