import numpy as np
from scipy import interpolate

def trim(spectrum_x: np.ndarray, spectrum_y: np.ndarray, reference_spectrum_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_spectrum_x = reference_spectrum_x.astype(float)
    mask = (spectrum_x >= reference_spectrum_x.min()) & (spectrum_x <= reference_spectrum_x.max())
    trimmed_x = spectrum_x[mask]
    trimmed_y = spectrum_y[mask]
    if len(trimmed_x) == 0:
        print("Нет пересечения диапазонов с библиотекой!")
        return np.array([]), np.array([])
    sort_idx = np.argsort(trimmed_x)
    return trimmed_x[sort_idx], trimmed_y[sort_idx]

def interpol(
        spectrum_x: np.ndarray, 
        spectrum_y: np.ndarray, 
        reference_spectrum_x: np.ndarray,
        ) -> np.ndarray:
    
    f = interpolate.interp1d(spectrum_x, spectrum_y, kind="cubic", bounds_error=False, fill_value=0.0)
    interpolated_y = f(reference_spectrum_x)
    return interpolated_y