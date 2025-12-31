import numpy as np
import rampy as rp
from create_dataframe import detect_separator


class Spectrum():
    def __init__(self, path):
        self.path = path
        delimiter = detect_separator(path)
        self.spectrum = np.genfromtxt(self.path, skip_header=10, delimiter=delimiter)
        self.x = self.spectrum[:, 0]
        self.y = self.spectrum[:, 1]
        self.name = ""

def process_lib_spectra(library: list, library_path: str) -> list:
    library_norm = []
    for lib_spec in library:
        try:
            lib_spectrum = Spectrum(f"{library_path}/{lib_spec}")  # Создаем экземпляр класса Spectrum
        except Exception as e:
            print(f"\nОшибка при чтении спектра: {lib_spec}")
            print(f"Причина: {e}\n")
            raise
        lib_corr_y, _ = rp.baseline(lib_spectrum.x, lib_spectrum.y, method="als", niter = 5) #Приводим к базовой линии
        lib_norm_y = rp.normalise(lib_corr_y.flatten(), lib_spectrum.x) #Нормализуем

        full_name = lib_spectrum.path.split("/")[1]
        if "_" in full_name:
            name = full_name.split("_")[0]
        else:
            name = full_name.split(".")[0]

        lib_spectrum.name = name
        lib_spectrum.y = lib_norm_y
        library_norm.append(lib_spectrum)

    return library_norm