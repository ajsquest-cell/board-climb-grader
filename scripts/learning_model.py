import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance



def scale_features(X, feature_columns):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X[feature_columns])
    return pd.DataFrame(scaled, columns=feature_columns, index=X.index)


def evaluate_two_bucket_histogram(df, feature_columns, threashold=19):
    bucket_df = df[df["climb_grade"].notna()].copy()

    # Two buckets:
    # 0 -> grade <= 19
    # 1 -> grade > 19
    bucket_df["bucket_label"] = (bucket_df["climb_grade"] > threashold).astype(int)

    X = bucket_df[feature_columns]
    y = bucket_df["bucket_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # HistGradientBoosting does NOT require feature scaling
    model = HistGradientBoostingClassifier(
        learning_rate=0.02,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)


    # Compute permutation importance
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="accuracy"
    )

    # Create a DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": result.importances_mean,
        "Std": result.importances_std
    })

    # Sort from most to least important
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    print("\nPermutation Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"{row['Feature']:30} {row['Importance']:.6f} ± {row['Std']:.6f}")

    print("\n2-Bucket Histogram Gradient Boosting Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def evaluate_two_bucket_SGDClassaifier(df, feature_columns, threshold=21):

    binary_df = df[df["climb_grade"].notna()].copy()
    binary_df["binary_label"] = (binary_df["climb_grade"] >= threshold).astype(int)

    X = binary_df[feature_columns]
    y = binary_df["binary_label"]

    # Split and scale as before
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = scale_features(X_train, feature_columns)
    X_test_scaled = scale_features(X_test, feature_columns)

    epochs = 100
    accuracy_history = []

    model = SGDClassifier(
        loss="log_loss",      # Logistic regression
        max_iter=1,           # One epoch at a time
        warm_start=True,      # Continue training each iteration
        random_state=42
    )

    classes = np.unique(y_train)

    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(X_train_scaled, y_train, classes=classes)
        else:
            model.partial_fit(X_train_scaled, y_train)

        # Training accuracy after this epoch
        train_pred = model.predict(X_train_scaled)
        accuracy_history.append(accuracy_score(y_train, train_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), accuracy_history, marker='o')
    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def evaluate_two_bucket_logistic(df, feature_columns, threshold=21):
    binary_df = df[df["climb_grade"].notna()].copy()
    binary_df["binary_label"] = (binary_df["climb_grade"] >= threshold).astype(int)

    X = binary_df[feature_columns]
    y = binary_df["binary_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=max(1000, len(df)))
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # Compute permutation importance
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="accuracy"
    )

    # Create a DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": result.importances_mean,
        "Std": result.importances_std
    })

    # Sort from most to least important
    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    print("\nPermutation Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"{row['Feature']:30} {row['Importance']:.6f} ± {row['Std']:.6f}")

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
    
    below = (binary_df["climb_grade"] < threshold).sum()
    above = (binary_df["climb_grade"] >= threshold).sum()

    print(f"Threshold {threshold}")
    print(f"Below {threshold}: {below}")
    print(f"{threshold} and above: {above}")

    true_positive = conf_matrix[1, 1]
    true_negative = conf_matrix[0, 0]

    plt.figure(figsize=(8, 5))
    ax = plt.bar(["True Negatives", "True Positives"], [true_negative, true_positive], color=["#4C78A8", "#54A24B"])
    plt.title("True Negatives vs True Positives")
    plt.ylabel("Count")
    for bar, value in zip(ax, [true_negative, true_positive]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(value), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

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

