from chainer import Chain
from src.models.models import iParaphraseNet, iParaphraseIoUNet
from src.models.phrase_only_models import (
    PhraseOnlyNet,
    WordEmbeddingAverage,
    LSTMPhraseEmbedding,
)


def build_multimodal_model(cfg) -> Chain:
    gate_mode: str = cfg.MODEL.SUB_NAME
    h_size: Tuple[int, int] = cfg.MODEL.H_SIZE

    if cfg.MODEL.USE_IOU:
        model = iParaphraseIoUNet(gate_mode, h_size)
    else:
        model = iParaphraseNet(gate_mode, h_size)
    return model


def build_phrase_model(cfg) -> Chain:
    sub_name: str = cfg.MODEL.SUB_NAME

    if sub_name == "wea":
        phrase_emb = WordEmbeddingAverage()
    elif sub_name == "lstm":
        phrase_emb = LSTMPhraseEmbedding(300)
    else:
        raise RuntimeError("invalid model type")

    model = PhraseOnlyNet(phrase_emb)

    return model


def build_model(cfg) -> Chain:
    if cfg.MODEL.NAME == "multimodal":
        model = build_multimodal_model(cfg)
    elif cfg.MODEL.NAME == "phrase-only":
        model = build_phrase_model(cfg)
    else:
        raise RuntimeError(f"invalid model cfg.MODEL.NAME")
    return model
