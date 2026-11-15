"""Endpointing: sign segment ends when both hands are 'at rest' for rest_s seconds."""
import numpy as np


def hand_speed(loc_seq):
    """loc_seq (T,42) body-frame location; speed = |delta wrist| for each hand."""
    d = np.diff(loc_seq, axis=0, prepend=loc_seq[:1])
    return np.linalg.norm(d[:, :3], axis=1) + np.linalg.norm(d[:, 21:24], axis=1)


class Endpointer:
    def __init__(self, fps=30, rest_s=0.8, speed_thr=0.004, min_active_s=0.3):
        self.rest_n = int(rest_s * fps); self.thr = speed_thr
        self.min_active = int(min_active_s * fps)
        self.reset()

    def reset(self):
        self.active = 0; self.rest = 0; self.prev = None; self.buf = []

    def push(self, loc, present=True):
        """Returns segment (list of frames) when an utterance ends, else None."""
        sp = 0.0 if self.prev is None or not present else float(np.abs(loc[[0,1,2,21,22,23]] - self.prev[[0,1,2,21,22,23]]).sum())
        self.prev = loc
        moving = present and sp > self.thr
        if moving:
            self.active += 1; self.rest = 0
        elif self.active:
            self.rest += 1
        if self.active:
            self.buf.append(loc)
        if self.active >= self.min_active and self.rest >= self.rest_n:
            seg = self.buf[:-self.rest_n] if self.rest_n else self.buf
            self.reset()
            return seg
        return None
