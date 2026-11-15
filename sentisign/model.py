import torch, torch.nn as nn
from .features import STREAM_DIMS


class TCNBlock(nn.Module):
    def __init__(s, c, d, p=0.2):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(c, c, 3, padding=d, dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(p))
    def forward(s, x): return x + s.net(x)


class StreamEncoder(nn.Module):
    def __init__(s, d_in, c=64):
        super().__init__(); s.proj = nn.Conv1d(d_in, c, 1)
        s.tcn = nn.Sequential(*[TCNBlock(c, d) for d in (1, 2, 4)])
    def forward(s, x): return s.tcn(s.proj(x.transpose(1, 2))).transpose(1, 2)  # (B,T,c)


class MultiStreamSign(nn.Module):
    """Per-stream TCN -> concat -> BiLSTM -> attention pooling -> classifier."""
    def __init__(s, n_cls, c=64, hid=128, streams=STREAM_DIMS):
        super().__init__()
        s.names = list(streams); s.dims = list(streams.values())
        s.enc = nn.ModuleList([StreamEncoder(d, c) for d in s.dims])
        s.lstm = nn.LSTM(c * len(s.dims), hid, 2, batch_first=True, bidirectional=True, dropout=0.2)
        s.att = nn.Linear(2 * hid, 1)
        s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hid, 256), nn.GELU(), nn.Linear(256, n_cls))
    def forward(s, x):
        parts = torch.split(x, s.dims, dim=-1)
        h = torch.cat([e(p) for e, p in zip(s.enc, parts)], -1)
        h, _ = s.lstm(h)
        w = torch.softmax(s.att(h), 1)
        return s.head((w * h).sum(1))


def augment(x, p=0.8):
    """Time-warp-free cheap aug: gaussian noise + random scale of location stream + temporal shift."""
    if torch.rand(1) > p: return x
    x = x + torch.randn_like(x) * 0.01
    sh = int(torch.randint(-3, 4, (1,)))
    return torch.roll(x, sh, 1)
