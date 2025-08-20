from spectrum import Spectrum
import matplotlib.pyplot as plt
import os
import rampy as rp


library_path = "spectra_new"
library = os.listdir(library_path) #Получаем список спектров в папке
for spec in library:
    lib_spec_path = f"{library_path}/{spec}"
    spectrum = Spectrum(lib_spec_path)
    sample_name = lib_spec_path.split("/")[1].split("_")[0]
    corrected_y, baseline = rp.baseline(spectrum.x, spectrum.y, method="als") #Приводим к базовой линии
    plt.figure(figsize=(10, 5))
    plt.plot(spectrum.x, spectrum.y, label='Original Spectrum')
    plt.plot(spectrum.x, baseline, label='Baseline', color='orange')
    plt.title(sample_name)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid()
    plt.show()