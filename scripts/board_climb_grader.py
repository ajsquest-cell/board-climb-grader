import re
import sqlite3
import pandas as pd
from pathlib import Path
import random

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import access_database as adb

# Alex King and Ben Hawkins
# board-climb-grader.py

PLACEMENTS_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"

## Board Constants
# x_min = -24
# x_max = 168
# y_min = 0
# y_max = 156
# board_width = x_max - x_min
# board_height = y_max - y_min

# Hold ID
HAND_HOLD_IDS = {1, 2, 3}
FOOT_HOLD_IDS = {4}
MIDDLE_HOLD_ID = 2

HOLD_MAPPING = {
    1: 'start',
    2: 'middle',
    3: 'finish',
    4: 'foot'
}

# Load hold placements
placement_coords = adb.extract_holes()

# defulted middle
def get_hold_type(hold_id):
    return HOLD_MAPPING.get(hold_id, 'middle')

# Example frame string: "p12r12p13r13p14r14p15r15"
def parse_frame(frame):
    if isinstance(frame, str) and frame.strip():
        numbers = re.findall(r"p(\d+)r(\d+)", frame)
        return [(int(p), int(r)) for p, r in numbers]
    else:
        return None

# Helper for testing  
def print_hold_data(hold_data):
    for hold in hold_data:
        print(f"Hold ID: {hold['hold_id']}, X: {hold['x']}, Y: {hold['y']}, Type: {hold['hold_type']}, Hand: {hold['is_hand_hold']}, Foot: {hold['is_foot_hold']}")

# Extract features from a given frame and angle
def extract_features_from_frame(angle, frame, is_nomatch, edge_left, edge_right, edge_bottom, edge_top):
    holds = parse_frame(frame)

    x_min = edge_left
    x_max = edge_right
    y_min = edge_bottom
    y_max = edge_top
    board_width = x_max - x_min
    board_height = y_max - y_min

    if holds is None:
        raise ValueError("Invalid frame format.")

    hold_data = []
    # p is placement id, h is hold id
    for p_id, h_id in holds:
        coordinate = placement_coords.get(p_id, (None, None))
        if coordinate[0] is None or coordinate[1] is None:
            continue

        hold_type = get_hold_type(h_id)
        is_hand_hold = h_id in HAND_HOLD_IDS
        is_foot_hold = h_id in FOOT_HOLD_IDS
        hold_data.append({
            'hold_id': p_id,
            'hold_type': hold_type,
            'x': coordinate[0],
            'y': coordinate[1],
            'is_hand_hold': is_hand_hold,
            'is_foot_hold': is_foot_hold
        })

    holds_df = pd.DataFrame(hold_data, columns=['hold_id', 'hold_type', 'x', 'y', 'is_hand_hold', 'is_foot_hold'])

    if holds_df.empty:
        return None

    hand_holds = holds_df[holds_df['is_hand_hold']]
    foot_holds = holds_df[holds_df['is_foot_hold']]
    start_holds = holds_df[holds_df['hold_type'] == 'start']
    finish_holds = holds_df[holds_df['hold_type'] == 'finish']
    middle_holds = holds_df[holds_df['hold_type'] == 'middle']

    x = holds_df['x'].to_numpy()
    y = holds_df['y'].to_numpy()

    features = {}

    #implemented features for testing:

    # Base features
    features['angle'] = angle
    features["angle_squared"] = float(angle) ** 2 if angle is not None else None
    features['num_holds'] = len(holds)
    features['num_hand_holds'] = len(hand_holds)
    features['num_foot_holds'] = len(foot_holds)
    features['start_holds'] = len(start_holds)
    features['finish_holds'] = len(finish_holds)
    features['middle_holds'] = len(middle_holds)
    features['is_nomatch'] = int(is_nomatch == 1)

    # Spatial features
    features['mean_y'] = float(np.mean(y))
    features['std_y'] = float(np.std(y)) if len(y) > 1 else 0.0
    features['std_X'] = float(np.std(x)) if len(x) > 1 else 0.0
    features['min_y'] = float(np.min(y))
    features['max_y'] = float(np.max(y))
    features['min_x'] = float(np.min(x))
    features['max_x'] = float(np.max(x))
    features['range_y'] = features['max_y'] - features['min_y']
    features['range_x'] = features['max_x'] - features['min_x']

    features['start_height'] = float(start_holds['y'].min()) if features['start_holds'] > 0 else None
    features['finish_height'] = float(finish_holds['y'].max()) if features['finish_holds'] > 0 else None
    features['height_gained'] = (
        (features['finish_height'] - features['start_height']) 
        if features['start_height'] is not None and features['finish_height'] is not None else None
        )
    features['hold_per_unit_height'] = float(features['num_holds'] / max(features['range_y'], 1)) if features['range_y'] > 0 else 0

    # Density features
    features['box_area'] = (
        float(features['range_x'] * features['range_y'])
        if features['range_x'] is not None and features['range_y'] is not None and features['range_x'] > 0 and features['range_y'] > 0 else 1
    )
    features['hold_density'] = float(features['num_holds'] / features['box_area']) if features['box_area'] else 0
    features['hand_hold_density'] = float(features['num_hand_holds'] / features['box_area']) if features['box_area'] else 0

    # Hand reach features
    if features['num_hand_holds'] >= 2:
        hand_pairs = hand_holds[['x', 'y']].to_numpy()
        hand_hold_distances = sp.spatial.distance.pdist(hand_pairs)
        hand_x = hand_holds['x'].to_numpy()
        hand_y = hand_holds['y'].to_numpy()

        features['mean_hand_reach'] = float(np.mean(hand_hold_distances))
        features['max_hand_reach'] = float(np.max(hand_hold_distances))
        features['std_hand_reach'] = float(np.std(hand_hold_distances))
        features['hand_spread_x'] = float(np.max(hand_x) - np.min(hand_x))
        features['hand_spread_y'] = float(np.max(hand_y) - np.min(hand_y))
    else:
        features['mean_hand_reach'] = 0.0
        features['max_hand_reach'] = 0.0
        features['std_hand_reach'] = 0.0
        features['hand_spread_x'] = 0.0
        features['hand_spread_y'] = 0.0

    # Foot reach features
    if features['num_hand_holds'] > 0 and features['num_foot_holds'] > 0:
        hand_pairs = hand_holds[['x', 'y']].to_numpy()
        foot_pairs = foot_holds[['x', 'y']].to_numpy()

        distances = []

        for hand_x, hand_y in hand_pairs:
            for foot_x, foot_y in foot_pairs:
                distance = float(np.sqrt((hand_x - foot_x) ** 2 + (hand_y - foot_y) ** 2))
                distances = np.append(distances, distance)
        distances = np.array(distances)

        features['mean_hand_foot_reach'] = float(np.mean(distances))
        features['max_hand_foot_reach'] = float(np.max(distances))
        features['min_hand_foot_reach'] = float(np.min(distances))
        features['std_hand_foot_reach'] = float(np.std(distances))
    else:
        features['mean_hand_foot_reach'] = 0.0
        features['max_hand_foot_reach'] = 0.0
        features['min_hand_foot_reach'] = 0.0
        features['std_hand_foot_reach'] = 0.0

    # Normalized features
    

    # Other features
    # features['hand_to_foot_ratio'] = features['num_hand_holds'] / features['num_foot_holds'] if features['num_foot_holds'] else float('inf')
    # features['angle_x_holds'] = features['angle'] * features['num_holds']

    return features

def evaluate_two_bucket_logistic(df, feature_columns, threshold=19):
    binary_df = df[df["climb_grade"].notna()].copy()
    binary_df["binary_label"] = (binary_df["climb_grade"] >= threshold).astype(int)

    X = binary_df[feature_columns]
    y = binary_df["binary_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=4000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n2-Bucket Logistic Regression Results:")
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')


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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n3-Bucket Random Forest Results:")
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

# Testing
def main():
    difficulty_grades = adb.extract_difficulty_grades()

    # number of climbs and offset
    count = 5000
    offset = random.randint(0, adb.get_table_size("climbs") - count)

    # Fetch rows from the database
    rows = adb.fetch_rows_from(count, offset)
    # rows = adb.fetch_random_rows(count)

    climb_data = []

    for row in rows:
        record = {
            'climb_id': row['uuid'],
            'angle': adb.fetch_angle(row['uuid']),
            'frames': row['frames'],
            'is_nomatch': row['is_nomatch'],
            'edge_left': row['edge_left'],
            'edge_right': row['edge_right'],
            'edge_bottom': row['edge_bottom'],
            'edge_top': row['edge_top'],
        }

        stats = adb.fetch_climb_stats(record['climb_id'])
        record['climb_grade'] = int(stats['display_difficulty']) if stats and stats.get('display_difficulty') is not None else None
        
        features = extract_features_from_frame(
            record['angle'], 
            record['frames'], 
            record['is_nomatch'], 
            record['edge_left'], 
            record['edge_right'], 
            record['edge_bottom'], 
            record['edge_top'])
        
        if features is None:
            continue

        record.update(features)
        climb_data.append(record)
        # Testing
        # print(f"Climb ID: {record['climb_id']}, Grade: {record['climb_grade']}, Features: {features}")

    df = pd.DataFrame(climb_data)

    # Remove rows with missing features
    df = df[df["climb_grade"].notna()]

    feature_columns = [
        "angle", 
        "angle_squared", 
        "num_holds", 
        "height_gained", 
        "hold_density", 
        "mean_hand_reach", 
        "max_hand_reach", 
        "std_hand_reach", 
        "hold_per_unit_height", 
        "is_nomatch", 
        "mean_hand_foot_reach", 
        "max_hand_foot_reach", 
        "min_hand_foot_reach", 
        "std_hand_foot_reach"
        ]
    # Fill NaN values in features with 0
    df[feature_columns] = df[feature_columns].fillna(0)

    # Keep the existing classifier on the original 10-28 grade range.
    classification_df = df[(df["climb_grade"] >= 10) & (df["climb_grade"] <= 28)].copy()
    X_class = classification_df[feature_columns]
    y_class = classification_df["climb_grade"]

    X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_class_train, y_class_train)

    y_class_pred = model.predict(X_class_test)
    accuracy = accuracy_score(y_class_test, y_class_pred)

    # Calculate within buckets
    within_1 = np.mean(np.abs(y_class_test - y_class_pred) <= 1)
    within_2 = np.mean(np.abs(y_class_test - y_class_pred) <= 2)

    conf_matrix = confusion_matrix(y_class_test, y_class_pred)

    print(f'Exact Accuracy: {accuracy:.4f}')
    print(f'Within 1 Grade Accuracy: {within_1:.4f}')
    print(f'Within 2 Grades Accuracy: {within_2:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    evaluate_two_bucket_logistic(df, feature_columns)
    evaluate_bucketed_random_forest(df, feature_columns)


if __name__ == "__main__":
    main()
