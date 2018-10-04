import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn.basic import LSTM, Layer
from nn.initialization import get_activation_by_name


class MaskedLSTM(LSTM):
    def forward(self, x_t, mask_t, hc_tm1):
        hc_t = super(MaskedLSTM, self).forward(x_t, hc_tm1)
        hc_t = mask_t * hc_t + (1 - mask_t) * hc_tm1

        return hc_t

    def forward_all_hl(self, x, mask, h0=None, return_c=False):
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

    def forward_all_doc(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out * 2,), dtype=theano.config.floatX)
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
