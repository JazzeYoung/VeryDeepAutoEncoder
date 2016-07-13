# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 09:00:27 2016

@author: jazze
"""

import numpy
import theano
import theano.tensor as T

import rbm
#from deeplearning import rbm

class DBN():

    def __init__(self, vsize=None, hsizes=[], lr=None, bsize=10, seed=123):
        assert vsize and hsizes and lr

        input = T.dmatrix('global_input')

        self.layers = []
        for hsize in hsizes:
            r = rbm.RBM(input=input, vsize=vsize, hsize=hsize, bsize=bsize,
                        lr=lr, seed=seed)
            self.layers.append(r)

            # configure inputs for subsequent layer
            input = self.layers[-1].hid
            vsize = hsize
