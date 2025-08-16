import numpy as np
from scipy import interpolate

def trim(spectrum_x: np.ndarray, spectrum_y: np.ndarray, library_spectrum_x: np.ndarray) -> np.ndarray:
    mask = (spectrum_x >= library_spectrum_x.min()) & (spectrum_x <= library_spectrum_x.max())
    trimmed_x = spectrum_x[mask]
    if len(trimmed_x) == 0:
        print("Нет пересечения диапазонов с библиотекой!")
        return

    index_min_x = np.where(spectrum_x == trimmed_x.min())
    index_max_x = np.where(spectrum_x == trimmed_x.max())
    trimmed_y = spectrum_y[index_min_x[0][0]:index_max_x[0][0]+1]
    return trimmed_x, trimmed_y

def interpol(
        spectrum_x: np.ndarray, 
        spectrum_y: np.ndarray, 
        library_spectrum_x: np.ndarray,
        ) -> np.ndarray:
    
    f = interpolate.interp1d(spectrum_x, spectrum_y, kind="cubic", bounds_error=False, fill_value=0.0)
    interpolated_y = f(library_spectrum_x)
    return interpolated_y