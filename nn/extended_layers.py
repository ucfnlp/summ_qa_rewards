import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn.advanced import RCNN
from nn.basic import LSTM
from nn.initialization import create_shared, random_init, sigmoid


class ExtRCNN(RCNN):
    def forward(self, x_t, mask_t, hc_tm1):
        hc_t = super(ExtRCNN, self).forward(x_t, hc_tm1)
        hc_t = mask_t * hc_t + (1 - mask_t) * hc_tm1
        return hc_t

    def forward_all(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out * (self.order + 1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x, mask],
            outputs_info=[h0]
        )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out * self.order:]
        else:
            return h[:, self.n_out * self.order:]

    def forward_all_2(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out * (self.order + 1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x, mask],
            outputs_info=[h0]
        )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out * self.order:]
        else:
            return h[:, self.n_out * self.order:]

    def copy_params(self, from_obj):
        self.internal_layers = from_obj.internal_layers
        self.bias = from_obj.bias


class ExtLSTM(LSTM):
    def forward(self, x_t, mask_t, hc_tm1):
        hc_t = super(ExtLSTM, self).forward(x_t, hc_tm1)
        hc_t = mask_t * hc_t + (1 - mask_t) * hc_tm1
        return hc_t

    def forward_all(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x, mask],
            outputs_info=[h0]
        )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out:]
        else:
            return h[:, self.n_out * self.order:]

    def forward_all_2(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x, mask],
            outputs_info=[h0]
        )

        return h
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out:]
        else:
            return h[:, self.n_out * self.order:]

    def copy_params(self, from_obj):
        self.internal_layers = from_obj.internal_layers


class HLLSTM(LSTM):
    def forward(self, x_t, mask_t, hc_tm1):
        hc_t = super(HLLSTM, self).forward(x_t, hc_tm1)
        hc_t = mask_t * hc_t + (1 - mask_t) * hc_tm1
        return hc_t

    def forward_all(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out *2,), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x, mask],
            outputs_info=[h0]
        )

        if return_c:
            return h[-1, :]
        elif x.ndim > 1:
            return h[-1 :, :, self.n_out:]
        else:
            return h[-1 :, self.n_out:]

    def forward_all_x(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out *2,), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x, mask],
            outputs_info=[h0]
        )

        if return_c:
            return h
        elif x.ndim > 1:
            return h[:, :, self.n_out:]
        else:
            return h[ :, self.n_out:]

    def copy_params(self, from_obj):
        self.internal_layers = from_obj.internal_layers


class LossComponent(object):
    def __init__(self, h_final_y):
        self.h_final_y = h_final_y

    def inner_argmin(self, gs, sys, min_prev):
        return ifelse(T.lt(min_prev, sys - gs), min_prev, sys - gs)

    def outer_sum(self, h_final, s):
        min_prev = T.zeros(h_final.shape, dtype=theano.config.floatX)
        return s + theano.scan(fn=self.inner_argmin, sequences=[self.h_final_y], non_sequences=[h_final, min_prev])

    def get_loss(self, h_final):
        sum_set = T.zeros(h_final.shape, dtype=theano.config.floatX)
        (s, _) = theano.scan(fn=self.outer_sum, sequences=[h_final], non_sequences=[sum_set])
        return s


class ZLayer(object):
    def __init__(self, n_in, n_hidden, activation, layer, test=False):
        self.n_in, self.n_hidden, self.activation, self.layer, self.test = \
            n_in, n_hidden, activation, layer, test
        self.MRG_rng = MRG_RandomStreams()
        self.create_parameters()

    def create_parameters(self):
        n_in, n_hidden = self.n_in, self.n_hidden
        activation = self.activation

        self.w1 = create_shared(random_init((n_in,)), name="w1")
        self.w2 = create_shared(random_init((n_hidden,)), name="w2")
        bias_val = random_init((1,))[0]
        self.bias = theano.shared(np.cast[theano.config.floatX](bias_val))

        if self.layer == 'lstm':
            rlayer = LSTM((n_in + 1), n_hidden, activation=activation)
        else:
            rlayer = RCNN((n_in + 1), n_hidden, activation=activation, order=2)

        self.rlayer = rlayer
        self.layers = [rlayer]

    def forward(self, x_t, z_t, h_tm1, pz_tm1):
        pz_t = sigmoid(
            T.dot(x_t, self.w1) +
            T.dot(h_tm1[:, -self.n_hidden:], self.w2) +
            self.bias
        )

        xz_t = T.concatenate([x_t, z_t.reshape((-1, 1))], axis=1)
        h_t = self.rlayer.forward(xz_t, h_tm1)

        # batch
        return h_t, pz_t

    def forward_all(self, x, z):
        assert x.ndim == 3
        assert z.ndim == 2
        xz = T.concatenate([x, z.dimshuffle((0, 1, "x"))], axis=2)
        h0 = T.zeros((1, x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        h = self.rlayer.forward_all(xz)
        h_prev = T.concatenate([h0, h[:-1]], axis=0)
        assert h.ndim == 3
        assert h_prev.ndim == 3
        pz = sigmoid(
            T.dot(x, self.w1) +
            T.dot(h_prev, self.w2) +
            self.bias
        )
        assert pz.ndim == 2
        return pz

    def sample(self, x_t, z_tm1, h_tm1):

        pz_t = sigmoid(
            T.dot(x_t, self.w1) +
            T.dot(h_tm1[:, -self.n_hidden:], self.w2) +
            self.bias
        )

        # batch
        pz_t = pz_t.ravel()

        if self.test:
            z_t = T.cast(T.ge(pz_t, 0.5), theano.config.floatX)
        else:
            z_t = T.cast(self.MRG_rng.binomial(size=pz_t.shape,
                                           p=pz_t), theano.config.floatX)

        xz_t = T.concatenate([x_t, z_t.reshape((-1, 1))], axis=1)
        h_t = self.rlayer.forward(xz_t, h_tm1)

        return z_t, h_t

    def sample_all(self, x):
        if self.layer == 'lstm':
            h0 = T.zeros((x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        else:
            h0 = T.zeros((x.shape[1], self.n_hidden * (self.rlayer.order + 1)), dtype=theano.config.floatX)

        z0 = T.zeros((x.shape[1],), dtype=theano.config.floatX)
        ([z, h], updates) = theano.scan(fn=self.sample, sequences=[x], outputs_info=[z0, h0])

        assert z.ndim == 2
        return z, updates

    @property
    def params(self):
        return [x for layer in self.layers for x in layer.params] + \
               [self.w1, self.w2, self.bias]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.w1.set_value(param_list[-3].get_value())
        self.w2.set_value(param_list[-2].get_value())
        self.bias.set_value(param_list[-1].get_value())
