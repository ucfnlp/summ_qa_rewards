import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn.advanced import RCNN
from nn.basic import LSTM, Layer
from nn.initialization import create_shared, random_init, sigmoid, get_activation_by_name


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


class SamplingLSTM(LSTM):
    def forward(self, x_t, h_tm1):
        hc_t = super(SamplingLSTM, self).forward(x_t, h_tm1)

        return hc_t

    def forward_all(self, x, y, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * (self.order + 1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
            fn=self.forward,
            sequences=[x],
            outputs_info=[h0]
        )
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


class Sampler(LSTM):
    def __init__(self, n_in, n_out, fc_in, fc_out, sample=False):
        super(Sampler, self).__init__(n_in=n_in, n_out=n_out)
        if sample:
            self.MRG_rng = MRG_RandomStreams()

        self.fc_layer = Layer(n_in=fc_in,
                              n_out=fc_out,
                              activation=get_activation_by_name('relu'),
                              has_bias=True)

        self.fc_layer_final = Layer(n_in=fc_out,
                                    n_out=1,
                                    activation=get_activation_by_name('sigmoid'),
                                    has_bias=True,
                                    clip_inp=True)

    def _forward(self, x_t, x_tm1, posit_x_t, mask_t, hc_tm1):
        x_tm1 = x_tm1 * mask_t
        hc_t = super(Sampler, self).forward(x_tm1, hc_tm1)

        concat_in = T.concatenate([x_t, posit_x_t, hc_t], axis=1)
        a_t = self.fc_layer.forward(concat_in)
        pt = self.fc_layer_final.forward(a_t)

        return hc_t, pt

    def _forward_inf(self, x_t, x_tm1, posit_x_t, mask_t, hc_tm1):
        x_tm1 = x_tm1 * mask_t
        hc_t = super(Sampler, self).forward(x_tm1, hc_tm1)

        concat_in = T.concatenate([x_t, posit_x_t, hc_t], axis=1)
        a_t = self.fc_layer.forward(concat_in)
        pt = self.fc_layer_final.forward(a_t)

        mask_next = T.cast(T.round(pt, mode='half_away_from_zero'), theano.config.floatX).reshape((x_t.shape[0],))
        return mask_next.dimshuffle((0, 'x')), hc_t, pt

    def _forward_sample(self, x_t, x_tm1, posit_x_t, mask_t, hc_tm1):
        x_tm1 = x_tm1 * mask_t
        hc_t = super(Sampler, self).forward(x_tm1, hc_tm1)

        concat_in = T.concatenate([x_t, posit_x_t, hc_t], axis=1)
        a_t = self.fc_layer.forward(concat_in)
        pt = self.fc_layer_final.forward(a_t).ravel()

        mask_next = T.cast(self.MRG_rng.binomial(size=pt.shape,
                                           p=pt), theano.config.floatX)
        mask_next = mask_next.reshape((-1,1))
        return mask_next, hc_t

    def pt_forward_all(self, x, posit_x, mask):
        h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)

        padded = T.shape_padaxis(T.zeros_like(x[0]), axis=1).dimshuffle((1, 0, 2))
        x_shifted = T.concatenate([padded, x[:-1]], axis=0)

        padded_mask = T.shape_padaxis(T.zeros_like(mask[0]), axis=1).dimshuffle((1, 0))
        mask = T.concatenate([padded_mask, mask[:-1]], axis=0).dimshuffle((0, 1, 'x'))

        o, _ = theano.scan(
            fn=self._forward,
            sequences=[x, x_shifted, posit_x, mask],
            outputs_info=[h0, None]
        )

        new_probs = o[1].reshape((x.shape[0], x.shape[1]))
        return new_probs

    def s_forward_all(self, x, posit_x, inference=False):
        h0 = T.zeros((x.shape[1], self.n_out * 2), dtype=theano.config.floatX)

        if not inference:
            return self._forward_all_sample(x, posit_x, h0)

        padded = T.shape_padaxis(T.zeros_like(x[0]), axis=1).dimshuffle((1, 0, 2))
        x_shifted = T.concatenate([padded, x[:-1]], axis=0)
        mask = T.zeros(shape=(x.shape[1],)).dimshuffle((0, 'x'))

        o, _ = theano.scan(
            fn=self._forward_inf,
            sequences=[x, x_shifted, posit_x],
            outputs_info=[mask, h0, None]
        )

        new_probs = o[2].reshape((x.shape[0], x.shape[1]))
        return new_probs

    def _forward_all_sample(self, x, posit_x, h0):
        padded = T.shape_padaxis(T.zeros_like(x[0]), axis=1).dimshuffle((1, 0, 2))
        x_shifted = T.concatenate([padded, x[:-1]], axis=0)
        mask = T.zeros(shape=(x.shape[1],)).dimshuffle((0, 'x'))

        [s, _], updates = theano.scan(
            fn=self._forward_sample,
            sequences=[x, x_shifted, posit_x],
            outputs_info=[mask, h0]
        )
        samples = theano.gradient.disconnected_grad(s).reshape((x.shape[0], x.shape[1]))
        padded_mask = T.shape_padaxis(T.zeros_like(samples[0]), axis=1).dimshuffle((1, 0))
        mask_from_samples = T.concatenate([padded_mask, samples[:-1]], axis=0).dimshuffle((0, 1, 'x'))

        [_, probs], _ = theano.scan(
            fn=self._forward,
            sequences=[x, x_shifted, posit_x, mask_from_samples],
            outputs_info=[h0, None]
        )

        return probs.reshape((x.shape[0], x.shape[1])), updates, samples

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

        z_t = T.cast(self.MRG_rng.binomial(size=pz_t.shape,
                                           p=pz_t), theano.config.floatX)

        xz_t = T.concatenate([x_t, z_t.reshape((-1, 1))], axis=1)
        h_t = self.rlayer.forward(xz_t, h_tm1)

        return z_t, h_t

    def sample_pretrain(self, x_t, z_tm1, h_tm1):

        pz_t = sigmoid(
            T.dot(x_t, self.w1) +
            T.dot(h_tm1[:, -self.n_hidden:], self.w2) +
            self.bias
        )

        # batch
        pz_t = pz_t.ravel()
        z_t = T.cast(T.round(pz_t, mode='half_away_from_zero'), theano.config.floatX)

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

    def sample_all_pretrain(self, x):
        if self.layer == 'lstm':
            h0 = T.zeros((x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        else:
            h0 = T.zeros((x.shape[1], self.n_hidden * (self.rlayer.order + 1)), dtype=theano.config.floatX)

        z0 = T.zeros((x.shape[1],), dtype=theano.config.floatX)
        ([z, h], updates) = theano.scan(fn=self.sample_pretrain, sequences=[x], outputs_info=[z0, h0])

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
