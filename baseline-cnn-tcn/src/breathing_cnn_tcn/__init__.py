"""CNN + TCN model: a per-frame CNN encoder feeding a dilated TCN decoder.

Pipeline stages
---------------
``clips``       Clip and session discovery.
``video``       Frame-exact grayscale decoding via an ffmpeg pipe.
``annotate``    Hand-place one crop box per session.
``targets``     Ground truth -> filtered, resampled, z-scored training target.
``channels``    Build the multi-channel per-frame input stack.
``preprocess``  Decode + crop + channelise every clip to a uint8 array on disk.
``dataset``     Windowed torch dataset over the stored arrays.
``model``       CNN frame encoder + dilated TCN, and the training loss.
``train``       Train on every non-reserved session.
``infer``       Whole-clip prediction by stitching overlapping windows.
``evaluate``    Score checkpoints on held-out clips via the real scorer, optionally
                writing the ``plot_diagnosis`` plots too (``--plot``).
``plot_diagnosis``  Rate-breakdown and reserved-grid diagnostic plots, called from
                ``evaluate --plot``.
"""

__all__ = [
    "annotate",
    "channels",
    "clips",
    "dataset",
    "evaluate",
    "infer",
    "model",
    "plot_diagnosis",
    "preprocess",
    "targets",
    "train",
    "video",
]
