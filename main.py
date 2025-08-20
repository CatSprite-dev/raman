import os
import joblib
import pandas as pd
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
    if not os.path.exists("model/logreg_model.joblib") or not os.path.exists("model/reference_df.csv"):
        library_path = os.environ.get("LIBRARY_PATH")
        model, reference_df = create_model(library_path)
    else:
        model = joblib.load("model/logreg_model.joblib")
        reference_df = pd.read_csv("model/reference_df.csv")

    df = create_dataframe_from_map(os.environ.get("MAP_PATH"))
    predictions = model.predict(df[[10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]])

    le = LabelEncoder() 
    le.fit(reference_df["target"])
    predicted_labels = le.inverse_transform(predictions)
    
    df['predicted_target'] = predicted_labels
    df.to_csv("predictions.csv", index=False)

    exclude = ['Epoxy', 'Tourmaline.txt', 'Tourmaline', "Brookite", "Epidote", "Titanite.txt"]  # список классов для исключения
    filtered = df[~df['predicted_target'].isin(exclude)]
    percentages = filtered['predicted_target'].value_counts(normalize=True) * 100
    print(percentages)

    


if __name__ == "__main__":
    main()