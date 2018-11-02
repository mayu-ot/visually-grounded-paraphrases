import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import reporter

def binary_classification_summary(y, t):
    xp = cuda.get_array_module(y)
    y = y.data

    y = y.ravel()
    true = t.ravel()
    pred = (y > 0)
    support = xp.sum(true)

    gtp_mask = xp.where(true)
    relevant = xp.sum(pred)
    tp = pred[gtp_mask].sum()

    if (support == 0) or (relevant == 0) or (tp == 0):
        return xp.array(0.), xp.array(0.), xp.array(0.)

    prec = tp * 1. / relevant
    recall = tp * 1. / support
    f1 = 2. * (prec * recall) / (prec + recall)

    return prec, recall, f1

class PhraseNet(chainer.Chain):
    def __init__(self, out_size):
        super(PhraseNet, self).__init__()
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
    
class GateNet(chainer.Chain):
    def __init__(self, out_size):
        super(GateNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.g_phr = L.Linear(None, out_size, initialW=w)
            self.g_img = L.Linear(None, out_size, initialW=w)
            
    def __call__(self, x_p, x_v):
        g_l = F.sigmoid(self.g_phr(x_p))
        g_v = F.sigmoid(self.g_img(x_v))
        return g_l, g_v

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
    
class MultiModalClassifierNet(chainer.Chain):
    def __init__(self, out_size):
        super(MultiModalClassifierNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l_phr = L.Linear(None, out_size, initialW=w)
            self.l_img = L.Linear(None, out_size, initialW=w)
            
            self.l_1 = L.Linear(None, out_size, initialW=w, nobias=True)
            self.bn_l0 = L.BatchNormalization(out_size)
            self.bn_v0 = L.BatchNormalization(out_size)
            self.bn_1 = L.BatchNormalization(out_size)
            
            self.cls = L.Linear(None, 1, initialW=w)

    def __call__(self, x_p, x_v):
        h = F.tanh(self.bn_l0(self.l_phr(x_p))) + F.tanh(self.bn_v0(self.l_img(x_v)))
        h = F.relu(self.bn_1(self.l_1(h)))
        h = self.cls(h)
        h = F.flatten(h)
        return h
    
class GatedClassifierNet(MultiModalClassifierNet):
    def __init__(self, out_size):
        super(GatedClassifierNet, self).__init__(out_size)
        w = initializers.HeNormal()
        with self.init_scope():
            self.gate_net = GateNet(out_size)

    def __call__(self, x_p, x_v):
        # gates
        g_l, g_v = self.gate_net(x_p, x_v)
        h_l = F.tanh(self.bn_l0(self.l_phr(x_p))) # added batchnormalization
        h_v = F.tanh(self.bn_v0(self.l_img(x_v))) # added batchnormalization
        h = g_l * h_l + g_v * h_v
        h = F.relu(self.bn_1(self.l_1(h)))
        h = self.cls(h)
        h = F.flatten(h)
        return h
    
class PhraseOnlyNet(chainer.Chain):
    def __init__(self):
        super(PhraseOnlyNet, self).__init__()
        with self.init_scope():
            self.phrase_net = PhraseNet(1000)
            self.classifier = SingleModalClassifierNet(300)
            
    def predict(self, phr_1, phr_2, xvis_1, xvis_2, l):
        _ = self(phr_1, phr_2, xvis_1, xvis_2, l)
        y = self.y
        return y
    
    def __call__(self, phr_1, phr_2, vis_1, vis_2, l):
        h = self.phrase_net(phr_1, phr_2)
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
    
class ImageOnlyNet(chainer.Chain):
    def __init__(self):
        super(ImageOnlyNet, self).__init__()
        with self.init_scope():
            self.image_net = ImgNet(1000)
            self.classifier = SingleModalClassifierNet(300)
            
    def predict(self, phr_1, phr_2, xvis_1, xvis_2, l):
        _ = self(phr_1, phr_2, xvis_1, xvis_2, l)
        y = self.y
        return y
    
    def __call__(self, phr_1, phr_2, vis_1, vis_2, l):
        h = self.image_net(vis_1, vis_2)
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
    
class Switching_iParaphraseNet(chainer.Chain):
    def __init__(self):
        super(Switching_iParaphraseNet, self).__init__()
        with self.init_scope():
            self.phrase_net = PhraseNet(1000)
            self.vision_net = ImgNet(1000)
            self.classifier = GatedClassifierNet(300)

    def predict(self, phr_1, phr_2, xvis_1, xvis_2, l):
        _ = self(phr_1, phr_2, xvis_1, xvis_2, l)
        y = self.y
        return y

    def __call__(self, phr_1, phr_2, vis_1, vis_2, l):
        h_p = self.phrase_net(phr_1, phr_2)
        h_v = self.vision_net(vis_1, vis_2)
        
        h = self.classifier(h_p, h_v)
        
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
    
    
class BaseNet(chainer.Chain):
    def __init__(self, h_size1, h_size2):
        super(BaseNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l_0 = L.Linear(None, h_size1, initialW=w, nobias=True)
            self.l_1 = L.Linear(None, h_size2, initialW=w, nobias=True)
            self.bn_0 = L.BatchNormalization(h_size1)
            self.bn_1 = L.BatchNormalization(h_size2)
            self.cls = L.Linear(None, 1)
    
    def __call__(self, x0, x1):
        h0 = F.relu(self.bn_0(self.l_0(x0)))
        h1 = F.relu(self.bn_0(self.l_0(x1)))
        
        h = F.relu(self.bn_1(self.l_1(h0) + self.l_1(h1)))
        h = self.cls(h)
        return h

class MultiModalGateNet(chainer.Chain):
    def __init__(self, h_size):
        super(MultiModalGateNet, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l_l0 = L.Linear(None, h_size, initialW=w, nobias=True)
            self.l_v0 = L.Linear(None, h_size, initialW=w, nobias=True)
            self.l= L.Linear(h_size, 1, initialW=w)
            
            self.bn_l = L.BatchNormalization(h_size)
            self.bn_v = L.BatchNormalization(h_size)
    
    def __call__(self, x_p1, x_p2, x_i1, x_i2):
        hl = F.relu(self.bn_l(self.l_l0(x_p1)+self.l_l0(x_p2)))
        hv = F.relu(self.bn_v(self.l_v0(x_i1)+self.l_v0(x_i2)))
        w = F.sigmoid(self.l(hl) + self.l(hv))
        return w
        
            
class LateSwitching_iParaphraseNet(chainer.Chain):
    def __init__(self):
        super(LateSwitching_iParaphraseNet, self).__init__()
        with self.init_scope():
            self.language_net = BaseNet(1000, 1000)
            self.vision_net = BaseNet(1000, 1000)
            self.gate_net = MultiModalGateNet(300)
            
    def __call__(self, phr_1, phr_2, vis_1, vis_2, l):
        y_l = self.language_net(phr_1, phr_2)
        y_v = self.vision_net(vis_1, vis_2)
        
        w = self.gate_net(phr_1, phr_2, vis_1, vis_2)
        y = w * y_l + (1-w)*y_v
        y = F.flatten(y)

        if chainer.config.train == False:
            self.y = F.sigmoid(y)
            self.t = l

        loss = F.sigmoid_cross_entropy(y, l)

        precision, recall, fbeta = binary_classification_summary(y, l)
        reporter.report({
            'loss': loss,
            'precision': precision,
            'recall': recall,
            'f1': fbeta
        }, self)

        return loss