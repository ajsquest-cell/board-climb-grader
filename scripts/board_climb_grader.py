import re
import sqlite3
import pandas as pd
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import access_database as adb

# Alex King and Ben Hawkins
# board-climb-grader.py

PLACEMENTS_PATH = Path(__file__).resolve().parent / "data" / "tb2.db"

## Board Constants
x_min = -24
x_max = 168
y_min = 0
y_max = 156
board_width = x_max - x_min
board_height = y_max - y_min

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
def extract_features_from_frame(angle, frame):
    holds = parse_frame(frame)

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
            'x': coordinate[0],
            'y': coordinate[1],
            'hold_type': hold_type,
            'is_hand_hold': is_hand_hold,
            'is_foot_hold': is_foot_hold
        })

    # testing
    # print_hold_data(hold_data)

    df = pd.DataFrame(hold_data)

    features = {}

    #implemented features for testing:

    # Base features
    features['angle'] = angle
    features["angle_squared"] = float(angle) ** 2 if angle is not None else None
    features['num_holds'] = len(holds)
    features['num_hand_holds'] = len([hold for hold in hold_data if hold['is_hand_hold']])
    features['num_foot_holds'] = len([hold for hold in hold_data if hold['is_foot_hold']])
    features['start_holds'] = len([hold for hold in hold_data if hold['hold_type'] == 'start'])
    features['finish_holds'] = len([hold for hold in hold_data if hold['hold_type'] == 'finish'])
    features['middle_holds'] = len([hold for hold in hold_data if hold['hold_type'] == 'middle'])

    # Spatial features
    if hold_data:
        features['mean_y'] = sum(hold['y'] for hold in hold_data) / len(hold_data)
        features['std_y'] = (sum((hold['y'] - features['mean_y']) ** 2 for hold in hold_data) / len(hold_data)) ** 0.5
        features['min_y'] = min(hold['y'] for hold in hold_data)
        features['max_y'] = max(hold['y'] for hold in hold_data)
        features['min_x'] = min(hold['x'] for hold in hold_data)
        features['max_x'] = max(hold['x'] for hold in hold_data)
        features['range_y'] = features['max_y'] - features['min_y']
        features['range_x'] = features['max_x'] - features['min_x']

        features['start_height'] = min(hold['y'] for hold in hold_data if hold['hold_type'] == 'start') if features['start_holds'] > 0 else None
        features['finish_height'] = max(hold['y'] for hold in hold_data if hold['hold_type'] == 'finish') if features['finish_holds'] > 0 else None
        features['height_gained_from_start_to_finish'] = (features['finish_height'] - features['start_height']) if features['start_height'] is not None and features['finish_height'] is not None else None
    else:
        features['mean_y'] = 0.0
        features['std_y'] = 0.0
        features['min_y'] = 0.0
        features['max_y'] = 0.0
        features['min_x'] = 0.0
        features['max_x'] = 0.0
        features['range_y'] = 0.0
        features['range_x'] = 0.0
        features['start_height'] = None
        features['finish_height'] = None
        features['height_gained_from_start_to_finish'] = None

    # Density features
    features['box_area'] = features['range_x'] * features['range_y'] if features['range_x'] is not None and features['range_y'] is not None and features['range_x'] > 0 and features['range_y'] > 0 else 1
    features['hold_density'] = features['num_holds'] / features['box_area'] if features['box_area'] else 0
    features['hold_density_squared'] = (features['num_holds'] / features['box_area']) ** 2 if features['box_area'] else 0
    features['hand_hold_density'] = features['num_hand_holds'] / features['box_area'] if features['box_area'] else 0
    features['foot_hold_density'] = features['num_foot_holds'] / features['box_area'] if features['box_area'] else 0
    # features['hold_per_unit_height'] = features['num_holds'] / features['height_gained'] if features['height_gained'] else 0

    # Hand reach features
    if features['num_hand_holds'] >= 2 and hold_data:
        hand_pairs = [i for i in range(1, len(hold_data)) if hold_data[i]['is_hand_hold'] and hold_data[i-1]['is_hand_hold']]
        if hand_pairs:
            features['mean_hand_reach'] = sum(abs(hold_data[i]['x'] - hold_data[i-1]['x']) for i in hand_pairs) / len(hand_pairs)
            features['max_hand_reach'] = float(max(abs(hold_data[i]['x'] - hold_data[i-1]['x']) for i in range(1, len(hold_data)) if hold_data[i]['is_hand_hold'] and hold_data[i-1]['is_hand_hold']))
            features['std_hand_reach'] = (sum((abs(hold_data[i]['x'] - hold_data[i-1]['x']) - features['mean_hand_reach']) ** 2 for i in range(1, len(hold_data)) if hold_data[i]['is_hand_hold'] and hold_data[i-1]['is_hand_hold']) / (features['num_hand_holds'] - 1)) ** 0.5
        else:
            features['mean_hand_reach'] = 0.0
            features['max_hand_reach'] = 0.0
            features['std_hand_reach'] = 0.0
        # features['hand_span_y'] = max(hold['y'] for hold in hold_data if hold['is_hand_hold']) - min(hold['y'] for hold in hold_data if hold['is_hand_hold'])
        # features['hand_span_x'] = max(hold['x'] for hold in hold_data if hold['is_hand_hold']) - min(hold['x'] for hold in hold_data if hold['is_hand_hold'])
    else:
        features['mean_hand_reach'] = 0.0
        features['max_hand_reach'] = 0.0
        features['std_hand_reach'] = 0.0
        # features['hand_span_y'] = 0.0
        # features['hand_span_x'] = 0.0

    # Other features
    # features['hand_to_foot_ratio'] = features['num_hand_holds'] / features['num_foot_holds'] if features['num_foot_holds'] else float('inf')
    # features['angle_x_holds'] = features['angle'] * features['num_holds']

    return features

# Testing
def main():
    difficulty_grades = adb.extract_difficulty_grades()

    # number of climbs and offset
    count = 1000
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
            'edge_left': row['edge_left'],
            'edge_right': row['edge_right'],
            'edge_bottom': row['edge_bottom'],
            'edge_top': row['edge_top'],
        }
        stats = adb.fetch_climb_stats(record['climb_id'])
        record['climb_grade'] = int(stats['display_difficulty']) if stats and stats.get('display_difficulty') is not None else None
        features = extract_features_from_frame(record['angle'], record['frames'])
        record.update(features)
        climb_data.append(record)
        # Testing
        # print(f"Climb ID: {record['climb_id']}, Grade: {record['climb_grade']}, Features: {features}")

    df = pd.DataFrame(climb_data)

    # Remove rows with missing features
    df = df[df["climb_grade"].notna()]

    # remove rows with outliers in features or difficulty
    # add code here
    
    # Fill NaN values in features with 0
    feature_columns = ["angle", "angle_squared", "num_holds", "height_gained_from_start_to_finish", "hold_density", "mean_hand_reach", "max_hand_reach", "std_hand_reach"]
    df[feature_columns] = df[feature_columns].fillna(0)

    X = df[feature_columns]
    y = df["climb_grade"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=len(climb_data))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate within buckets
    within_1 = np.mean(np.abs(y_test - y_pred) <= 1)
    within_2 = np.mean(np.abs(y_test - y_pred) <= 2)

    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Exact Accuracy: {accuracy:.4f}')
    print(f'Within 1 Grade Accuracy: {within_1:.4f}')
    print(f'Within 2 Grades Accuracy: {within_2:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')


if __name__ == "__main__":
    main()
