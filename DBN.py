import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
import rbm
#from rbm import test_rbm
#from deeplearning import rbm

class DBN():

    def __init__(self, vsize=None, hsizes=[], lr=None, bsize=10, seed=123):
        assert vsize and hsizes and lr

        #input = T.dmatrix('global_input')

        self.layers = []
        for hsize in hsizes:
            r = rbm.test_rbm(learning_rate=lr, output_folder='dbn_rbm_plots')

            # configure inputs for subsequent layer
            input = self.layers[-1].hid
            vsize = hsize

def test_dbn(pretrain_epoch=3, training_epochs=10, 
            batch_size = 300, dataset='mnist.pkl.gz'):
             
    hsizes = [100,200,300,400,500]
    print hsizes
    DBN(vsize=784, hsizes=[100,200,300,400,500], lr=0.01, bsize=10, seed=123)
    
    #dbn.test_DBN(pretraining_epochs, training_epochs, batch_size)
    
if __name__ == '__main__':
    test_dbn()