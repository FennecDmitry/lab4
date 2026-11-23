"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучаем ансамбль (голосование) и возвращаем метрики."""
    # Переводим в numpy
    X_train = np.array(data.train_features)
    y_train = np.array(data.train_target)
    X_test = np.array(data.test_features)
    y_test = np.array(data.test_target)

    # Создаем модели для ансамбля
    # Ограничиваем глубину, чтобы избежать переобучения
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=data.random_seed)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=data.random_seed)

    # Объединяем в ансамбль (мягкое голосование по вероятностям)
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )

    # Обучение только на train
    ensemble.fit(X_train, y_train)

    # Получаем вероятности и предсказания на test
    probabilities = ensemble.predict_proba(X_test)[:, 1]
    predictions = ensemble.predict(X_test)

    # Считаем метрики
    roc_auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    return EnsembleResult(
        model_name="Voting (RandomForest + GradientBoosting)",
        probabilities=probabilities.tolist(),
        predictions=predictions.tolist(),
        roc_auc=float(roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )