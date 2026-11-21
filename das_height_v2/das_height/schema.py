"""Common data schema shared by simulated and real (FEBUS) recordings."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DASRecord:
    """One walking pass recorded on one fibre section.

    data         : array (n_time, n_channel), strain or strain-rate
    fs           : temporal sampling rate per channel [Hz]
    dx           : channel spacing [m]
    gauge_length : gauge length [m]
    x0           : position of channel 0 along the fibre [m]
    units        : 'strain' or 'strain_rate'
    labels       : dict with at least subject_id, pass_id, height_cm
                   (height only for labelled training data)
    source       : 'sim' or 'febus'
    """
    data: np.ndarray
    fs: float
    dx: float
    gauge_length: float
    x0: float = 0.0
    units: str = "strain"
    labels: dict = field(default_factory=dict)
    source: str = "sim"

    @property
    def positions(self):
        return self.x0 + np.arange(self.data.shape[1]) * self.dx

    def to_strain_rate(self):
        """Return a copy in strain-rate units, so sim and FEBUS data match."""
        if self.units == "strain_rate":
            return self
        d = np.gradient(self.data, 1.0 / self.fs, axis=0)
        return DASRecord(d, self.fs, self.dx, self.gauge_length, self.x0,
                         "strain_rate", dict(self.labels), self.source)
