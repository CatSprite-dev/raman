import rampy as rp
import numpy as np
import enum
import matplotlib.pyplot as plt
import os
from create_dataframe import detect_separator


class Method(enum.Enum):
    rubberband = "rubberband"
    poly = "poly"
    als = "als"



class Spectrum():
    def __init__(self, path):
        self.path = path
        delimiter = detect_separator(path)
        self.spectrum = np.genfromtxt(self.path, skip_header=10, delimiter=delimiter)
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

    def substract_baseline(self, method, polynomial_order=None):
        if method == Method.rubberband:
            corrected_signal, baseline = rp.baseline(self.x, self.y, method="rubberband")
            return corrected_signal, baseline
        if method == Method.als:
            corrected_signal, baseline = rp.baseline(self.x, self.y, method="als", niter = 5)
            return corrected_signal, baseline
        if method == Method.poly:
            corrected_signal, baseline = rp.baseline(self.x, self.y, method="poly")
            return corrected_signal, baseline
        
    def show_all_spectra(self, spectra, src_dir):
        for spec in spectra:
            spectrum = Spectrum(os.path.abspath(f"{src_dir}/{spec}"))
            spectrum.create_plot(spectrum.x, spectrum.y, name=spec)
        self.show_plot()
    