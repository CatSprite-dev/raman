import pandas as pd
import joblib
from create_dataframe import create_reference_dataframe
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


def create_model(library_path):
    reference_df = create_reference_dataframe(library_path)
    le = LabelEncoder()
    reference_df["target_encoded"] = le.fit_transform(reference_df["target"])
    
    # Используем только нужные колонки для обучения модели
    X = reference_df[[10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]]
    y = reference_df['target_encoded']
    #X = reference_df.drop(['target', 'target_encoded'], axis=1)
    #y = reference_df['target_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.1, 
                                                    random_state = 42)
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(X_train, y_train)

    joblib.dump(model, "model/logreg_model.joblib")

    y_pred = model.predict(X_test)
    model_matrix = confusion_matrix(y_test, y_pred)
    model_matrix_df = pd.DataFrame(model_matrix)
    model_accuracy = accuracy_score(y_test, y_pred)
    return model, reference_df



