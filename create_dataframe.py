import pandas as pd
import numpy as np
import rampy as rp
import os
import joblib
from process import trim, interpol
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




def detect_separator(file_path: str) -> str:
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#") == False:
                if "\t" in line:
                    return "\t"
                if "," in line:
                    return ","
        

def get_spectrum_range(spectrum_path: str) -> tuple[np.ndarray, np.ndarray]:
    load_dotenv()
    
    #Создаем референсный спектр на сетку которого потом будем обрезать все остальные, он нужен только для обрезки
    #Позже надо будет заменить этот момент на что-то более умное, например, на обрезку по средним значениям
    reference_df = pd.read_csv(spectrum_path, comment='#', names=['W', 'I'])
    reference_spectrum = np.array(reference_df, dtype=float)
    reference_spectrum_x = reference_spectrum[:, 0][1:]
    reference_spectrum_y = reference_spectrum[:, 1][1:]

    return reference_spectrum_x, reference_spectrum_y


def create_reference_dataframe(library_path: str) -> pd.DataFrame:
    range_x, range_y = get_spectrum_range(os.environ.get("REFERENCE_SPECTRUM")) #Создаем референсный спектр

    library = os.listdir(library_path) #Получаем список спектров в папке

    general_data_frame = pd.DataFrame()
    targets = []
    for spec in library:
        lib_spec_path = f"{library_path}/{spec}"
        sample_name = lib_spec_path.split("/")[1].split("_")[0]
        sep = detect_separator(lib_spec_path)
        df = pd.read_csv(f"{library_path}/{spec}", comment='#', names=['W', 'I'], sep=sep) #Создаем df спектра
        spectrum = np.array(df) #Переводи его в np массив для дальнейшей работы
        spectrum_x = spectrum[:, 0]
        spectrum_y = spectrum[:, 1]

        corrected_y, baseline = rp.baseline(spectrum_x, spectrum_y, method="als", niter = 5) #Приводим к базовой линии
        corrected_y: np.ndarray = corrected_y.flatten() #Приводим 2D массив к 1D массиву так как .baseline() возвращает коррекцию в формате 2D (n, 1)

        trimmed_x, trimmed_y = trim(spectrum_x, corrected_y, range_x) #Обрезаем спектр до спектра эталона          
        interpolated_y = interpol(trimmed_x, trimmed_y, range_x) #Интерполируем спектр на сетку эталона

        temp_df = pd.DataFrame(interpolated_y.reshape(1, -1), columns=range_x) #Создаем временный df 
        targets.append(sample_name)
        general_data_frame = pd.concat([general_data_frame, temp_df], ignore_index=True) #Добавляем временный df к основному


    #Создаем объект StandardScaler для стандартизации данных
    scaler = StandardScaler()

    #Приводим данные к единому масштабу
    scaled_data = scaler.fit(general_data_frame)
    joblib.dump(scaler, 'model/scaler_model.pkl')  # Сохраняем scaler в файл
    scaled_data = scaler.transform(general_data_frame)

    general_data_frame_scaled = pd.DataFrame(scaled_data, columns=general_data_frame.columns)    

    # Создаем объект PCA с 2 компонентами
    pca = PCA(n_components=8)

    # Применяем к данным
    X_pca = pca.fit(general_data_frame_scaled)
    joblib.dump(pca, 'model/pca_model.pkl')
    X_pca = pca.transform(general_data_frame_scaled)

    #добавляем главные компоненты в основной датафрейм
    for i in range(X_pca.shape[1]):
        general_data_frame_scaled[i+10001] = X_pca[:, i]
    
    general_data_frame_scaled['target'] = targets  # Добавляем целевую переменную обратно в датафрейм

    #Сохраняем датафрейм в csv файл
    general_data_frame_scaled.to_csv("model/reference_df.csv", index=False)

    return general_data_frame_scaled

def create_dataframe_from_map(map_path: str):
    range_x, range_y = get_spectrum_range(os.environ.get("REFERENCE_SPECTRUM")) #Создаем референсный спектр

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
    """
    normalised_spectra = []
    for corr_spectrum in corrected_spectra:
        x = corr_spectrum[:, 0] #Первый столбик W
        y = corr_spectrum[:, 1] #Второй столбик I
        normalise_y = rp.normalise(y, x) #Нормализуем спектры, чтобы интенсивность была в пределах от 0 до 1
        normalised_spectra.append(np.column_stack((x, normalise_y))) #Создаем двумерный np массив из одномерных массивов x и normalise_y
    """
    smoothed_spectra = []
    for morm_spectrum in corrected_spectra: #Сглаживаем спектры, используя метод smooth и проделывая те же самые процедуры, что и в циклах выше
        x = morm_spectrum[:, 0]
        y = morm_spectrum[:, 1]
        smooothed_y = rp.smooth(x, y, method="savgol", window_length = 11, polyorder = 5)
        smoothed_spectra.append(np.column_stack((x, smooothed_y)))
    
    final_data_frame = pd.DataFrame()
    for smoothed_spectrum in smoothed_spectra:
        x = smoothed_spectrum[:, 0]
        y = smoothed_spectrum[:, 1]
        trimmed_x, trimmed_y = trim(x, y, range_x) #Обрезаем спектр до спектра эталона          
        interpolated_y = interpol(trimmed_x, trimmed_y, range_x) #Интерполируем спектр на сетку эталона

        temp_df = pd.DataFrame(interpolated_y.reshape(1, -1), columns=range_x) #Создаем временный df 
        final_data_frame = pd.concat([final_data_frame, temp_df], ignore_index=True) #Добавляем временный df к основному

    #Создаем объект StandardScaler для стандартизации данных
    scaler: StandardScaler = joblib.load("model/scaler_model.pkl")  # Загружаем scaler из файла

    #Приводим данные к единому масштабу
    scaled_data = scaler.transform(final_data_frame)
    final_data_frame_scaled = pd.DataFrame(scaled_data, columns=final_data_frame.columns)
    

    # Создаем объект PCA с 8 компонентами
    pca: PCA = joblib.load("model/pca_model.pkl")  # Загружаем pca из файла

    # Применяем к данным
    X_pca = pca.transform(final_data_frame_scaled)

    #добавим главные компоненты в основной датафрейм
    for i in range(X_pca.shape[1]):
        final_data_frame_scaled[i+10001] = X_pca[:, i]

    return final_data_frame_scaled


