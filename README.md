
# Raman Spectroscopic Data Analysis

This project is designed for the systematic analysis and classification of Raman spectroscopic data employing modern machine learning techniques. The framework provides tools for preprocessing, dimensionality reduction, and predictive modeling, with the ultimate goal of facilitating reproducible research in spectroscopic data science.

## Project Structure

```

.
├── main.py              # Primary script for executing the analysis
├── model.py             # Machine learning model implementation
├── create\_dataframe.py  # Construction of DataFrames from raw data
├── process.py           # Spectral preprocessing functions
├── spectrum.py          # Class for managing and visualizing spectra
├── requirements.txt     # Project dependencies
└── model/               # Directory for storing trained models

````

## Installation

1. Clone the repository.  
2. Install the required dependencies:
```bash
pip install -r requirements.txt
````

## Usage

### Running the analysis

```bash
python main.py
```

### Environment configuration

Define the necessary paths by creating a `.env` file in the root directory of the project with the following variables:

```
LIBRARY_PATH=path_to_spectrum_library
MAP_PATH=path_to_spectral_map
REFERENCE_SPECTRUM=path_to_reference_spectrum
```

## Functionality

### Core modules:

1. **main.py** – orchestrates the overall workflow:

   * Loads or trains a classification model
   * Processes new spectroscopic measurements
   * Generates predictions
   * Exports results to a CSV file

2. **model.py** – implements and trains a multinomial logistic regression model for spectral classification

3. **create\_dataframe.py** – transforms spectroscopic datasets into structured DataFrames suitable for analysis

4. **process.py** – provides preprocessing functionality:

   * Spectral trimming
   * Spectral interpolation

5. **spectrum.py** – a dedicated class for handling individual spectra:

   * Visualization of Raman spectra
   * Baseline correction (noted to be computationally intensive)
   * Spectral comparison

## Features

* Dimensionality reduction via Principal Component Analysis (PCA)
* Classification using multinomial logistic regression
* Support for multiple baseline correction strategies
* Integrated visualization tools for Raman spectra

## Output

The framework produces a `predictions.csv` file containing the predicted class labels for each spectrum and reports the class distribution in percentages.

## Requirements

* Python 3.7+
* Dependencies as specified in `requirements.txt`



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
