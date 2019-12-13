import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import initializers
from chainer import reporter
from chainer import variable
from chainer import function

class BinaryClassificationSummary(function.Function):
    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

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

def binary_classification_summary(y, t):
    return BinaryClassificationSummary()(y, t)

def b_embed_text(texts, W):
    embs = [F.average(F.embed_id(t, W, ignore_label=-1), axis=0) for t in texts]
    embs = F.stack(embs)
    return embs

class FuseNet(chainer.Chain):
    def __init__(self, h_size):
        super(FuseNet, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, h_size)
        
    def __call__(self, v, p):
        h = F.concat([F.normalize(v), p], axis=-1)
        h = self.fc(h)
        return h
    
class VGPNet(chainer.Chain):
    def __init__(self, h_size, emb_W):
        super(VGPNet, self).__init__()
        with self.init_scope():
            self.emb = variable.Parameter(emb_W)
            self.fuse_net = FuseNet(h_size)
            self.fc = L.Linear(None, 128)
            self.cls = L.Linear(None, 1)
    
    def predict(self, p0, p1, v0, v1):
        p0 = b_embed_text(p0, self.emb)
        p1 = b_embed_text(p1, self.emb)
        
        h0 = self.fuse_net(v0, p0)
        h1 = self.fuse_net(v1, p1)
        h = F.concat([h0, h1], axis=-1)
        h = F.relu(self.fc(h))
        h = self.cls(h)
        y = F.sigmoid(h)
        return y
        
    def __call__(self, p0, p1, v0, v1, l):
        p0 = b_embed_text(p0, self.emb)
        p1 = b_embed_text(p1, self.emb)
        
        h0 = self.fuse_net(v0, p0)
        h1 = self.fuse_net(v1, p1)
        h = F.concat([h0, h1], axis=-1)
        h = F.relu(self.fc(h))
        h = F.flatten(self.cls(h))
        
        loss = F.sigmoid_cross_entropy(h, l)
        _, _, f1 = binary_classification_summary(h, l)
        reporter.report({'loss': loss, 'f1': f1}, self)
        
        return loss