import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

def main():
    print("Загрузка данных...")
    # Загружаем датасет (разделитель - точка с запятой)
    df = pd.read_csv("german.csv", sep=';')
    
    # Первый столбец - целевая переменная (Creditability)
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # Разбиваем на train и test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Масштабируем данные (очень важно для таких моделей)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Формируем данные для контракта
    input_data = EnsembleInput(
        train_features=X_train_scaled,
        train_target=y_train.values,
        test_features=X_test_scaled,
        test_target=y_test.values,
        random_seed=42
    )

    print("Обучение ансамбля...")
    result = run_ensemble_experiment(input_data)

    print("\n--- РЕЗУЛЬТАТЫ ---")
    print(f"Модель: {result.model_name}")
    print(f"ROC-AUC:  {result.roc_auc:.4f}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision:{result.precision:.4f}")
    print(f"Recall:   {result.recall:.4f}")
    print(f"F1-score: {result.f1:.4f}")

    # Сохраняем в JSON
    json_data = {
        "model_name": result.model_name,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
    
    print("\nРезультаты сохранены в experiment_results.json")

if __name__ == "__main__":
    main()