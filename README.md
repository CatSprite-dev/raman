
# Raman Spectroscopic Data Analysis

A project for analyzing and classifying Raman spectroscopic data using machine learning.

## Project Structure

```

.
├── main.py              # Main script for running the analysis
├── model.py             # Machine learning model
├── create_dataframe.py  # Create a DataFrame from data
├── process.py           # Spectrum processing functions
├── spectrum.py          # Class for working with spectra
├── requirements.txt     # Project dependencies
└── model/               # Directory for saving trained models

````

## Installation

1. Clone the repository  
2. Install dependencies:
```bash
pip install -r requirements.txt
````

## Usage

### Run the main script

```bash
python main.py
```

### Environment setup

Create a `.env` file in the project root directory with the following variables:

```
LIBRARY_PATH=path_to_spectrum_library
MAP_PATH=path_to_spectrum_map
REFERENCE_SPECTRUM=path_to_reference_spectrum
```

## Functionality

### Main modules:

1. **main.py** – main script that:

   * Loads or creates a classification model
   * Processes new data
   * Makes predictions
   * Saves results to a CSV file

2. **model.py** – builds and trains a multinomial logistic regression model for spectral classification

3. **create\_dataframe.py** – creates a DataFrame from spectroscopic data

4. **process.py** – functions for spectrum processing:

   * Trimming spectra
   * Spectrum interpolation

5. **spectrum.py** – class for working with individual spectra:

   * Spectrum visualization
   * Baseline correction (slow)
   * Spectrum comparison

## Features

* Dimensionality reduction using Principal Component Analysis (PCA)
* Classification with logistic regression
* Support for multiple baseline correction methods
* Visualization of spectroscopic data

## Output

The program generates a `predictions.csv` file with predictions for each spectrum and displays the class distribution statistics in percentages.

## Requirements

* Python 3.7+
* Dependencies listed in `requirements.txt`

================================================================================================================================================

# Анализ спектроскопических данных Рамана

Проект для анализа и классификации спектроскопических данных Рамана с использованием машинного обучения.

## Структура проекта

```
.
├── main.py              # Основной скрипт для запуска анализа
├── model.py             # Модель машинного обучения
├── create_dataframe.py  # Создание DataFrame из данных
├── process.py           # Функции обработки спектров
├── spectrum.py          # Класс для работы со спектрами
├── requirements.txt     # Зависимости проекта
└── model/               # Директория для сохранения моделей
```

## Установка

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

### Основной запуск
```bash
python main.py
```

### Настройка окружения
Создайте файл `.env` в корневой директории проекта со следующими переменными:
```
LIBRARY_PATH=путь_к_библиотеке_спектров
MAP_PATH=путь_к_карте_спектров
REFERENCE_SPECTRUM=путь_к_эталонному_спектру
```

## Функциональность

### Основные модули:

1. **main.py** - основной скрипт, который:
   - Загружает или создает модель классификации
   - Обрабатывает новые данные
   - Выполняет предсказания
   - Сохраняет результаты в CSV файл

2. **model.py** - создает и обучает модель многоклассовой логистической регрессии для классификации спектров

3. **create_dataframe.py** - создает DataFrame из спектроскопических данных

4. **process.py** - функции для обработки спектров:
   - Обрезка спектров (trim)
   - Интерполяция спектров (interpol)

5. **spectrum.py** - класс для работы с отдельными спектрами:
   - Визуализация спектров
   - Коррекция базовой линии (работет медленно)
   - Сравнение спектров

## Особенности

- Использование метода главных компонент (PCA) для уменьшения размерности
- Классификация с помощью логистической регрессии
- Поддержка различных методов коррекции базовой линии
- Визуализация спектроскопических данных

## Выходные данные

Программа создает файл `predictions.csv` с предсказаниями для каждого спектра и выводит статистику распределения классов в процентах.

## Требования

- Python 3.7+
- Зависимости указаны в requirements.txt
```
