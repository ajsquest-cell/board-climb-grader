import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

def scale_features(X, feature_columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X[feature_columns])
    return pd.DataFrame(scaled, columns=feature_columns, index=X.index)


def evaluate_two_bucket_logistic(df, feature_columns, threshold=19):
    binary_df = df[df["climb_grade"].notna()].copy()
    binary_df["binary_label"] = (binary_df["climb_grade"] >= threshold).astype(int)

    X = binary_df[feature_columns]
    y = binary_df["binary_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = scale_features(X_train, feature_columns)
    X_test_scaled = scale_features(X_test, feature_columns)

    model = LogisticRegression(max_iter=max(1000, len(df)))
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1]
    )

    print("\n2-Bucket Logistic Regression Results:")
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print("Class Metrics:")
    for label, p, r, f in zip([0, 1], precision, recall, f1):
        print(f'  Class {label}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}')

    
    coef_abs = np.abs(model.coef_[0])
    feature_importance = pd.Series(coef_abs, index=feature_columns).sort_values(ascending=False)

    top_features = feature_importance.head(12)

    plt.figure(figsize=(14, 7))
    ax = top_features.plot(kind='bar', color='steelblue', width=0.8)
    ax.set_title('Top Logistic Regression Feature Importances', fontsize=14)
    ax.set_ylabel('Absolute Coefficient Magnitude', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', labelsize=10)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    plt.tight_layout()
    plt.show()


def evaluate_bucketed_random_forest(df, feature_columns):
    bucket_df = df[df["climb_grade"].notna()].copy()

    def bucket_grade(grade):
        if grade <= 17:
            return 0
        elif grade <= 23:
            return 1
        else:
            return 2

    bucket_df["bucket_label"] = bucket_df["climb_grade"].apply(bucket_grade)

    X = bucket_df[feature_columns]
    y = bucket_df["bucket_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = scale_features(X_train, feature_columns)
    X_test_scaled = scale_features(X_test, feature_columns)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n3-Bucket Random Forest Results:")
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')


def evaluate_all_grades_logistic(df, feature_columns):
    # Keep the existing classifier on the original 10-28 grade range.
    classification_df = df[(df["climb_grade"] >= 10) & (df["climb_grade"] <= 28)].copy()
    X_class = classification_df[feature_columns]
    y_class = classification_df["climb_grade"]

    x_class_train, x_class_test, y_class_train, y_class_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    X_class_train_scaled = scale_features(x_class_train, feature_columns)
    X_class_test_scaled = scale_features(x_class_test, feature_columns)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_class_train_scaled, y_class_train)

    y_class_pred = model.predict(X_class_test_scaled)
    accuracy = accuracy_score(y_class_test, y_class_pred)

    # Calculate within buckets
    within_1 = np.mean(np.abs(y_class_test - y_class_pred) <= 1)
    within_2 = np.mean(np.abs(y_class_test - y_class_pred) <= 2)

    conf_matrix = confusion_matrix(y_class_test, y_class_pred)

    print(f'Exact Accuracy: {accuracy:.4f}')
    print(f'Within 1 Grade Accuracy: {within_1:.4f}')
    print(f'Within 2 Grades Accuracy: {within_2:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

