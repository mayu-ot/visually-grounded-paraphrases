from typing import Tuple
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import reporter
from .phrase_only_models import WordEmbeddingAverage, PhraseTransformNet
from dataclasses import dataclass


def binary_classification_summary(y, t):
    xp = cuda.get_array_module(y)
    y = y.data

    y = y.ravel()
    true = t.ravel()
    pred = y > 0
    support = xp.sum(true)

    gtp_mask = xp.where(true)
    relevant = xp.sum(pred)
    tp = pred[gtp_mask].sum()

    if (support == 0) or (relevant == 0) or (tp == 0):
        return xp.array(0.0), xp.array(0.0), xp.array(0.0)

    prec = tp * 1.0 / relevant
    recall = tp * 1.0 / support
    f1 = 2.0 * (prec * recall) / (prec + recall)

    return prec, recall, f1


class ImgNet(chainer.Chain):
    def __init__(self, out_size):
        super(ImgNet, self).__init__()
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


@dataclass()
class ClassifierNet(chainer.Chain):
    out_size: int

    def __post_init__(self):
        super(ClassifierNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l_phr = L.Linear(None, self.out_size, initialW=w)
            self.l_img = L.Linear(None, self.out_size, initialW=w)

            self.l_1 = L.Linear(None, self.out_size, initialW=w, nobias=True)
            self.bn_l0 = L.BatchNormalization(self.out_size)
            self.bn_v0 = L.BatchNormalization(self.out_size)
            self.bn_1 = L.BatchNormalization(self.out_size)

            self.cls = L.Linear(None, 1, initialW=w)

    def __call__(self, x_p, x_v):
        h_l = F.tanh(self.bn_l0(self.l_phr(x_p)))
        h_v = F.tanh(self.bn_v0(self.l_img(x_v)))
        h = h_l * 0.5 + h_v * 0.5
        h = F.relu(self.bn_1(self.l_1(h)))
        h = self.cls(h)
        h = F.flatten(h)
        return h


@dataclass()
class GatedClassifierNet(ClassifierNet):
    gate_net: chainer.Chain

    def __post_init__(self):
        super(GatedClassifierNet, self).__post_init__()
        with self.init_scope():
            self.gate = self.gate_net

    def __call__(self, x_p, x_v):
        g_l, g_v = self.gate(x_p, x_v)
        h_l = F.tanh(self.bn_l0(self.l_phr(x_p)))
        h_v = F.tanh(self.bn_v0(self.l_img(x_v)))
        h = g_l * h_l + g_v * h_v
        h = F.relu(self.bn_1(self.l_1(h)))
        h = self.cls(h)
        h = F.flatten(h)
        return h


@dataclass()
class GateNet(chainer.Chain):
    out_size: int
    mode: str

    def __post_init__(self):
        super(GateNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.g_phr = L.Linear(None, self.out_size, initialW=w)
            self.g_img = L.Linear(None, self.out_size, initialW=w)

    def __call__(self, *args):
        if self.mode == "language_gate":
            x, _ = args  # use only the first modality
        elif self.mode == "visual_gate":
            _, x = args
        elif self.mode == "multimodal_gate":
            x = F.concat(args, axis=1)
        else:
            raise RuntimeError("invaild gate mode")

        g_l = F.sigmoid(self.g_phr(x))
        g_v = F.sigmoid(self.g_img(x))
        return g_l, g_v


@dataclass
class iParaphraseNet(chainer.Chain):
    gate_mode: str
    h_size: Tuple[int, int] = (1000, 300)

    def __post_init__(self):
        super(iParaphraseNet, self).__init__()
        with self.init_scope():
            self.phrase_emb = WordEmbeddingAverage()
            self.phrase_net = PhraseTransformNet(self.h_size[0])
            self.vision_net = ImgNet(self.h_size[0])

            if self.gate_mode == "none":
                classifier = ClassifierNet(self.h_size[1])
            else:
                gate_net = GateNet(self.h_size[1], self.gate_mode)
                classifier = GatedClassifierNet(self.h_size[1], gate_net)

            self.classifier = classifier

    def predict(self, phr_a, phr_b, xvis_a, xvis_b, l):
        _ = self(phr_a, phr_b, xvis_a, xvis_b, l)
        y = self.y
        return y

    def __call__(self, phr_a, phr_b, vis_a, vis_b, l):
        emb_a = self.phrase_emb(phr_a)
        emb_b = self.phrase_emb(phr_b)
        h_p = self.phrase_net(emb_a, emb_b)
        h_v = self.vision_net(vis_a, vis_b)

        h = self.classifier(h_p, h_v)

        if chainer.config.train is False:
            self.y = F.sigmoid(h)
            self.t = l

        loss = F.sigmoid_cross_entropy(h, l)

        precision, recall, fbeta = binary_classification_summary(h, l)
        reporter.report(
            {
                "loss": loss,
                "precision": precision,
                "recall": recall,
                "f1": fbeta,
            },
            self,
        )

        return loss


# class NaiveFuse_iParaphraseNet(Switching_iParaphraseNet):
#     def __init__(self):
#         super(NaiveFuse_iParaphraseNet, self).__init__()

#     def setup_layers(self, _):
#         with self.init_scope():
#             self.phrase_net = PhraseNet(1000)
#             self.vision_net = ImgNet(1000)
#             self.classifier = ClassifierNet(300)


# class BaseNet(chainer.Chain):
#     def __init__(self, h_size1, h_size2):
#         super(BaseNet, self).__init__()
#         w = initializers.HeNormal()
#         with self.init_scope():
#             self.l_0 = L.Linear(None, h_size1, initialW=w, nobias=True)
#             self.l_1 = L.Linear(None, h_size2, initialW=w, nobias=True)
#             self.bn_0 = L.BatchNormalization(h_size1)
#             self.bn_1 = L.BatchNormalization(h_size2)
#             self.cls = L.Linear(None, 1)

#     def __call__(self, x0, x1):
#         h0 = F.relu(self.bn_0(self.l_0(x0)))
#         h1 = F.relu(self.bn_0(self.l_0(x1)))

#         h = F.relu(self.bn_1(self.l_1(h0) + self.l_1(h1)))
#         h = self.cls(h)
#         return h


# class LateSwitching_iParaphraseNet(chainer.Chain):
#     def __init__(self):
#         super(LateSwitching_iParaphraseNet, self).__init__()
#         with self.init_scope():
#             self.language_net = BaseNet(1000, 300)
#             self.vision_net = BaseNet(1000, 300)
#             self.gate_net = MultiModalGateNet(1)

#     def __call__(self, phr_1, phr_2, vis_1, vis_2, l):
#         y_l = self.language_net(phr_1, phr_2)
#         y_v = self.vision_net(vis_1, vis_2)

#         g_l, g_v = self.gate_net(phr_1, phr_2, vis_1, vis_2)
#         y = g_l * y_l + g_v * y_v
#         y = F.flatten(y)

#         if chainer.config.train == False:
#             self.y = F.sigmoid(y)
#             self.t = l

#         loss = F.sigmoid_cross_entropy(y, l)

#         precision, recall, fbeta = binary_classification_summary(y, l)
#         reporter.report(
#             {
#                 "loss": loss,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": fbeta,
#             },
#             self,
#         )

#         return loss
