import numpy as np
from spectrum import Spectrum, Method
import os

src_dir = "spectra"

spectra = os.listdir(src_dir)
print(spectra)
print(type(spectra))
print(os.path.abspath(spectra[0]))
for spec in spectra:
    spectrum = Spectrum(os.path.abspath(f"{src_dir}/{spec}"))
    spectrum.create_plot(spectrum.x, spectrum.y, name=spec)
spectrum.show_plot()

def main():
    spectrum = Spectrum("spectra/Quartz_532__RAW.txt")
    spectrum.create_plot(spectrum.x, spectrum.y, name="Original", color="black")

    for method in Method:
        corr, base = spectrum.substract_baseline(method)
        spectrum.create_plot(spectrum.x, corr, name=method.name)

    spectrum.show_plot()


if __name__ == "__main__":
    main()
