import rampy as rp
import numpy as np
import enum
import matplotlib.pyplot as plt


class Method(enum.Enum):
    rubberband = "rubberband"
    poly = "poly"
    arPLS = "arPLS"



class Spectrum():
    def __init__(self, path):
        self.path = path
        self.spectrum = np.genfromtxt(self.path, skip_header=10, delimiter=',')
        self.x = self.spectrum[:, 0]
        self.y = self.spectrum[:, 1]

    def create_plot(self, x, y, name=None, color=None):
        plt.plot(x, y, label=name, color=color)
        plt.xlabel("Raman shift, cm-1")
        plt.ylabel("Intensity")
        plt.title(self.path.split("/")[-1])
        plt.legend()
    
    def show_plot(self, fig=None):
        plt.show()

    def substract_baseline(self, method):
        if method == Method.rubberband:
            corrected_signal, baseline = rp.baseline(self.x, self.y, method="rubberband")
            return corrected_signal, baseline
        if method == Method.arPLS:
            corrected_signal, baseline = rp.baseline(self.x, self.y, method="arPLS", lam=1e3, ratio=0.05)
            return corrected_signal, baseline
        if method == Method.poly:
            corrected_signal, baseline = rp.baseline(self.x, self.y, method="poly")
            return corrected_signal, baseline
    