import os
from dotenv import load_dotenv
from create_dataframe import create_dataframe_from_map
from model import create_model
from sklearn.preprocessing import LabelEncoder




def main():   
    load_dotenv()
    library_path = os.environ.get("LIBRARY_PATH")
    model, reference_df = create_model(library_path)

    df = create_dataframe_from_map("trash/rm2442-1.2.txt")
    predictions = model.predict(df)

    le = LabelEncoder() 
    le.fit(reference_df["target"])
    predicted_labels = le.inverse_transform(predictions)
    
    df['predicted_target'] = predicted_labels
    df.to_csv("predictions.csv", index=False)

    


if __name__ == "__main__":
    main()