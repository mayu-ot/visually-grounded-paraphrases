import chainer
import chainer.links as L
import chainer.functions as F

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