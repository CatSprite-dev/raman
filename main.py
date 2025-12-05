import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from create_dataframe import create_dataframe_from_map
from model import create_model
from sklearn.preprocessing import LabelEncoder
import time



def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Время выполнения {func.__name__}: {end - start:.2f} секунд")
        return result
    return wrapper
    

@timeit
def main():   
    load_dotenv()
    
    # Создаем папку model если её нет
    os.makedirs("model", exist_ok=True)
    
    if not os.path.exists("model/logreg_model.joblib") or not os.path.exists("model/reference_df.csv") or not os.path.exists("model/label_encoder.joblib"):
        print("Создание и обучение модели...")
        library_path = os.environ.get("LIBRARY_PATH")
        model, reference_df = create_model(library_path)
        le = joblib.load("model/label_encoder.joblib")
    else:
        print("Загрузка обученной модели...")
        model = joblib.load("model/logreg_model.joblib")
        reference_df = pd.read_csv("model/reference_df.csv")
        le = joblib.load("model/label_encoder.joblib")


    print("Обработка карты...")
    df: pd.DataFrame = create_dataframe_from_map(os.environ.get("MAP_PATH"))
    
    print("Предсказание классов...")
    predictions = model.predict(df)

    # Используем обученный LabelEncoder
    predicted_labels = le.inverse_transform(predictions)
    
    df['predicted_target'] = predicted_labels
    df.to_csv("predictions.csv", index=False)
    print("Предсказания сохранены в predictions.csv")

    # Анализ результатов
    print("\nСтатистика предсказаний:")
    total_predictions = len(predicted_labels)
    unique_classes, counts = np.unique(predicted_labels, return_counts=True)
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total_predictions) * 100
        print(f"{cls}: {count} спектров ({percentage:.2f}%)")

    # Исключение определенных классов (опционально)
    exclude = ['Anatase', 'Chromite', 'Pyrite', "Garnet", "Zircon", "Rutile", "Apatite"]
    if exclude:
        print(f"\nСтатистика после исключения классов {exclude}:")
        filtered = df[df['predicted_target'].isin(exclude)]
        filtered_total = len(filtered)
        
        if filtered_total > 0:
            filtered_classes, filtered_counts = np.unique(filtered['predicted_target'], return_counts=True)
            for cls, count in zip(filtered_classes, filtered_counts):
                percentage = (count / filtered_total) * 100
                print(f"{cls}: {count} спектров ({percentage:.2f}%)")
        else:
            print("Нет спектров после фильтрации")

    
if __name__ == "__main__":
    main()