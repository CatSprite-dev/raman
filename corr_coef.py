import os
import numpy as np
import rampy as rp
import pandas as pd
import matplotlib.pyplot as plt
from spectrum import Spectrum
from process import trim, interpol
from dotenv import load_dotenv

def main():
    load_dotenv()
    map_path = os.environ.get("MAP_PATH")
    df = pd.read_csv(map_path, sep="\t") #создаем DataFrame pandas
    grouped = df.groupby(['X', 'Y']) #Группируем строки по уникальным значениям X и Y
    
    spectra = []
    coordinates = []
    for coord, group in grouped:
        spectrum = group[['W', 'I']].values #Создаём np массив вида [[W1, I1], [W2, I2], ...]
        spectrum_sorted = spectrum[spectrum[:, 0].argsort()] #Сортируем массив
        spectra.append(spectrum_sorted)
        coordinates.append(coord)
        
    corrected_spectra = []
    baselines = []
    for spectrum in spectra:
        x = spectrum[:, 0] #Первый столбик W
        y = spectrum[:, 1] #Второй столбик I
        correct_y, baseline = rp.baseline(x, y, method="als", niter = 5) #Приводим спектр к базовой линии 
        corrected_spectra.append(np.column_stack((x, correct_y))) #Создаем двумерный массив np из одномерных массивов x и y(correcr_y)
        baselines.append(np.column_stack((x, baseline))) #То же самое и для базовой линии
    
    normalised_spectra = []
    for spectrum in corrected_spectra:
        x = spectrum[:, 0] #Первый столбик W
        y = spectrum[:, 1] #Второй столбик I
        normalise_y = rp.normalise(y, x) #Нормализуем спектры, чтобы интенсивность была в пределах от 0 до 1
        normalised_spectra.append(np.column_stack((x, normalise_y))) #Создаем двумерный np массив из одномерных массивов x и normalise_y
    
    smoothed_spectra = []
    for spectrum in normalised_spectra: #Сглаживаем спектры, используя метод smooth и проделывая те же самые процедуры, что и в циклах выше
        x = spectrum[:, 0]
        y = spectrum[:, 1]
        smooothed_y = rp.smooth(x, y, method="savgol", window_length = 11, polyorder = 5)
        smoothed_spectra.append(np.column_stack((x, smooothed_y)))
    
    library_path = os.environ.get("LIBRARY_PATH")
    library = os.listdir(library_path) #Получаем список спектров в папке
    common_count = 0
    for lib_spec in library:
        count = 0
        lib_spectrum = Spectrum(f"{library_path}/{lib_spec}") #Создаем экземпяр класса Spectrum
        lib_corr_y, _ = rp.baseline(lib_spectrum.x, lib_spectrum.y, method="als", niter = 5) #Приводим к базовой линии
        lib_norm_y = rp.normalise(lib_corr_y.flatten(), lib_spectrum.x) #Нормализуем
        for spec in smoothed_spectra:
            trimmed_x, trimmed_y = trim(spec[:, 0], spec[:, 1], lib_spectrum.x) #Обрезаем спеткр до диапазона библиотечного
            interpolated_y = interpol(trimmed_x, trimmed_y, lib_spectrum.x) #Интерполируем обрезанный спектр на библиотечный
            corr_coef = np.around(np.corrcoef(interpolated_y, lib_norm_y)[0,1], 2) #Высчитываем коэфициент корреляции
            if corr_coef > 0.7:
                count += 1
                common_count += 1
        
            name = lib_spectrum.path.split("/")[1].split("_")[0]
            if name == "Pyrite" and corr_coef > 0.7:
                plt.figure(figsize=(12, 6))
                plt.plot(lib_spectrum.x, interpolated_y, label = 'Спектр')
                plt.plot(lib_spectrum.x, lib_norm_y, label = 'Библиотечный')
                plt.title(f"Коэфициент корреляции = {corr_coef}")
                plt.legend()
                #plt.show()                

        print(f"{count/len(spectra)*100:.2f} % - {count} - {name}")
    print(f"Общее количество спектров = {common_count}")
        
        




if __name__ == "__main__":
    main()
