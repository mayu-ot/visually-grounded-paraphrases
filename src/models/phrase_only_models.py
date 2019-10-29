import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import reporter
from .utils import binary_classification_summary


class WordEmbeddingAverage(chainer.Chain):
    def __init__(self):
        super(WordEmbeddingAverage, self).__init__()
        w_arr = np.load('data/processed/word2vec.npy')
        with self.init_scope():
            self.w_emb = L.EmbedID(w_arr.shape[0],
                                   w_arr.shape[1],
                                   initialW=w_arr,
                                   ignore_label=-1)

    def __call__(self, indices):
        x = [F.average(self.w_emb(i), axis=0, keepdims=True) for i in indices]
        x = F.concat(x, axis=0)
        return x


class LSTMPhraseEmbedding(WordEmbeddingAverage):
    def __init__(self, out_size):
        super(LSTMPhraseEmbedding, self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1, 300, out_size, .0)

    def __call__(self, indices):
        x = [self.w_emb(i) for i in indices]
        hy, _, _ = self.lstm(None, None, x)
        hy = F.squeeze(hy)
        return hy


class PhraseTransformNet(chainer.Chain):
    def __init__(self, out_size):
        super(PhraseTransformNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l_0 = L.Linear(None, out_size, initialW=w, nobias=True)
            self.l_1 = L.Linear(None, out_size, initialW=w, nobias=True)
            self.bn_0 = L.BatchNormalization(out_size)
            self.bn_1 = L.BatchNormalization(out_size)

    def __call__(self, x0, x1):

        h0 = F.relu(self.bn_0(self.l_0(x0)))
        h1 = F.relu(self.bn_0(self.l_0(x1)))

        h = F.relu(self.bn_1(self.l_1(h0) + self.l_1(h1)))
        return h


class SingleModalClassifierNet(chainer.Chain):

    def __init__(self, out_size):
        super(SingleModalClassifierNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l_0 = L.Linear(None, out_size, initialW=w)
            self.l_1 = L.Linear(None, out_size, initialW=w, nobias=True)
            self.bn_0 = L.BatchNormalization(out_size)
            self.bn_1 = L.BatchNormalization(out_size)
            self.cls = L.Linear(None, 1, initialW=w)

    def __call__(self, x):
        h = F.tanh(self.bn_0(self.l_0(x)))
        h = F.relu(self.bn_1(self.l_1(h)))
        h = self.cls(h)
        h = F.flatten(h)
        return h


class PhraseOnlyNet(chainer.Chain):
    def __init__(self, phrase_emb):
        super(PhraseOnlyNet, self).__init__()
        with self.init_scope():
            self.phrase_emb = phrase_emb
            self.transform_net = PhraseTransformNet(1000)
            self.classifier = SingleModalClassifierNet(300)

    def predict(self, phr_a, phr_b, l):
        _ = self(phr_a, phr_b, l)
        y = self.y
        return y

    def __call__(self, phr_a, phr_b, l):
        phr_a = self.phrase_emb(phr_a)
        phr_b = self.phrase_emb(phr_b)
        h = self.transform_net(phr_a, phr_b)
        h = self.classifier(h)

        if chainer.config.train == False:
            self.y = F.sigmoid(h)
            self.t = l

        loss = F.sigmoid_cross_entropy(h, l)

        precision, recall, fbeta = binary_classification_summary(h, l)
        reporter.report({
            'loss': loss,
            'precision': precision,
            'recall': recall,
            'f1': fbeta
        }, self)

        return loss
