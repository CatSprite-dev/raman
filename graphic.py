from spectrum import Spectrum
import matplotlib.pyplot as plt


"""
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
"""
spectrum = Spectrum("library_2/Corundum__R060020-4__Raman__514__45__ccw__Raman_Data_Processed__18571.txt")
plt.figure(figsize=(10, 5))
plt.plot(spectrum.x, spectrum.y, label='Corundum')
plt.legend()
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.grid()
plt.show()