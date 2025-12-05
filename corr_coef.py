import os
import time
import numpy as np
import rampy as rp
import pandas as pd
import matplotlib.pyplot as plt
from spectrum import Spectrum
from process import trim, interpol
from dotenv import load_dotenv
from create_dataframe import detect_separator

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Время выполнения {func.__name__}: {end - start:.2f} секунд")
        return result
    return wrapper

def validate_dataframe_of_map(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = {"#X": "X", 
                   "Unnamed: 1": "Y",
                   "#Y": "W",
                   "Unnamed: 3": "I"}
    df = df.rename(columns=new_columns)
    return df.dropna(axis=1, how='all')


@timeit
def main():
    load_dotenv()
    map_path = os.environ.get("MAP_PATH")
    sep = detect_separator(map_path)
    df = validate_dataframe_of_map(pd.read_csv(map_path, sep=sep)) #создаем DataFrame pandas и валидируем его
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
        try:
            lib_spectrum = Spectrum(f"{library_path}/{lib_spec}")  # Создаем экземпляр класса Spectrum
        except Exception as e:
            print(f"\nОшибка при чтении спектра: {lib_spec}")
            print(f"Причина: {e}\n")
            raise

        #lib_spectrum = Spectrum(f"{library_path}/{lib_spec}") #Создаем экземпяр класса Spectrum
        lib_corr_y, _ = rp.baseline(lib_spectrum.x, lib_spectrum.y, method="als", niter = 5) #Приводим к базовой линии
        lib_norm_y = rp.normalise(lib_corr_y.flatten(), lib_spectrum.x) #Нормализуем
        for spec in smoothed_spectra:
            trimmed_x, trimmed_y = trim(spec[:, 0], spec[:, 1], lib_spectrum.x) #Обрезаем спеткр до диапазона библиотечного
            interpolated_y = interpol(trimmed_x, trimmed_y, lib_spectrum.x) #Интерполируем обрезанный спектр на библиотечный
            corr_coef = np.around(np.corrcoef(interpolated_y, lib_norm_y)[0,1], 2) #Высчитываем коэфициент корреляции
            
            full_name = lib_spectrum.path.split("/")[1]
            if "_" in full_name:
                name = full_name.split("_")[0]
            else:
                name = full_name.split(".")[0]

            if corr_coef > 0.5:
                if name.lower() in ["titanite.txt", "brookite.txt", "magnetite.txt"]:
                    if corr_coef > 0.65:
                        count += 1
                        common_count += 1
                else:
                    count += 1
                    common_count += 1
            
        
            """
            if name == "Pyrite" and corr_coef > 0.4:
                plt.figure(figsize=(12, 6))
                plt.plot(lib_spectrum.x, interpolated_y, label = 'Спектр')
                plt.plot(lib_spectrum.x, lib_norm_y, label = 'Библиотечный')
                plt.title(f"Коэфициент корреляции = {corr_coef}")
                plt.legend()
                #plt.show()       
            """         
        if count > 0:
            print(f"{count/len(spectra)*100:.2f} % - {count} - {name}")
    print(f"Общее количество спектров = {common_count}")
        
        




if __name__ == "__main__":
    main()
