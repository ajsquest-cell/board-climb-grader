import re
from pathlib import Path
import random

import pandas as pd
import numpy as np
import scipy as sp

import access_database as adb
import learning_model as lm

# Alex King and Ben Hawkins
# board-climb-grader.py

# Hold ID
HAND_HOLD_IDS = {1, 2, 3}
FOOT_HOLD_IDS = {4}
MIDDLE_HOLD_ID = {2}

HOLD_MAPPING = {
    1: 'start',
    2: 'middle',
    3: 'finish',
    4: 'foot'
    }


# Load hold placements dictionary
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
    try:
        angle_value = float(angle)
    except (TypeError, ValueError):
        angle_value = 0.0

    features['angle'] = angle_value
    features["angle_squared"] = angle_value ** 2
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
    features['board_area'] = (
        float(features['range_x'] * features['range_y'])
        if features['range_x'] is not None and features['range_y'] is not None and features['range_x'] > 0 and features['range_y'] > 0 else 1
    )
    features['hold_density'] = float(features['num_holds'] / features['board_area']) if features['board_area'] else 0
    features['hand_hold_density'] = float(features['num_hand_holds'] / features['board_area']) if features['board_area'] else 0
    features['foot_hold_density'] = float(features['num_foot_holds'] / features['board_area']) if features['board_area'] else 0
    features['hand_to_foot_density_ratio'] = (
        float(features['hand_hold_density'] / features['foot_hold_density'])
        if features['foot_hold_density'] and np.isfinite(features['hand_hold_density'] / features['foot_hold_density'])
        else 0.0
    )

    # Hand reach features
    if features['num_hand_holds'] >= 2:
        hand_pairs = hand_holds[['x', 'y']].to_numpy()
        hand_hold_distances = sp.spatial.distance.pdist(hand_pairs)
        
        hand_x = hand_holds['x'].to_numpy()
        hand_y = hand_holds['y'].to_numpy()

        features['mean_hand_reach'] = float(np.mean(hand_hold_distances))
        features['max_hand_reach'] = float(np.max(hand_hold_distances))
        features['min_hand_reach'] = float(np.min(hand_hold_distances))
        features['std_hand_reach'] = float(np.std(hand_hold_distances))
        features['hand_spread_x'] = float(np.max(hand_x) - np.min(hand_x))
        features['hand_spread_y'] = float(np.max(hand_y) - np.min(hand_y))
    else:
        features['mean_hand_reach'] = 0.0
        features['max_hand_reach'] = 0.0
        features['min_hand_reach'] = 0.0
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
    board_width_norm = max(board_width, 1)
    board_height_norm = max(board_height, 1)
    board_area_norm = board_width_norm * board_height_norm
    board_span_norm = max(board_width_norm, board_height_norm, 1)

    def safe_divide(numerator, denominator):
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.divide(numerator, denominator)
        return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    features['norm_mean_y'] = safe_divide(features['mean_y'], board_height_norm)
    features['norm_std_y'] = safe_divide(features['std_y'], board_height_norm)
    features['norm_std_x'] = safe_divide(features['std_X'], board_width_norm)
    features['norm_min_y'] = safe_divide(features['min_y'], board_height_norm)
    features['norm_max_y'] = safe_divide(features['max_y'], board_height_norm)
    features['norm_min_x'] = safe_divide(features['min_x'], board_width_norm)
    features['norm_max_x'] = safe_divide(features['max_x'], board_width_norm)
    features['norm_range_y'] = safe_divide(features['range_y'], board_height_norm)
    features['norm_range_x'] = safe_divide(features['range_x'], board_width_norm)
    
    features['norm_start_height'] = safe_divide(features['start_height'], board_height_norm) if features['start_height'] is not None else 0.0
    features['norm_finish_height'] = safe_divide(features['finish_height'], board_height_norm) if features['finish_height'] is not None else 0.0
    features['norm_height_gained'] = safe_divide(features['height_gained'], board_height_norm) if features['height_gained'] is not None else 0.0
    
    features['norm_board_area'] = safe_divide(features['board_area'], board_area_norm) if features['board_area'] is not None else 0.0
    features['norm_hold_density'] = safe_divide(features['num_holds'], board_area_norm)
    features['norm_hand_hold_density'] = safe_divide(features['num_hand_holds'], board_area_norm)
    features['norm_foot_hold_density'] = safe_divide(features['num_foot_holds'], board_area_norm)

    features['norm_mean_hand_reach'] = safe_divide(features['mean_hand_reach'], board_width_norm) if features['mean_hand_reach'] is not None else 0.0
    features['norm_max_hand_reach'] = safe_divide(features['max_hand_reach'], board_width_norm) if features['max_hand_reach'] is not None else 0.0
    features['norm_min_hand_reach'] = safe_divide(features['min_hand_reach'], board_width_norm) if features['min_hand_reach'] is not None else 0.0
    features['norm_std_hand_reach'] = safe_divide(features['std_hand_reach'], board_width_norm) if features['std_hand_reach'] is not None else 0.0
    features['norm_hand_spread_x'] = safe_divide(features['hand_spread_x'], board_width_norm) if features['hand_spread_x'] is not None else 0.0
    features['norm_hand_spread_y'] = safe_divide(features['hand_spread_y'], board_height_norm) if features['hand_spread_y'] is not None else 0.0
    
    features['norm_mean_hand_foot_reach'] = safe_divide(features['mean_hand_foot_reach'], board_span_norm) if features['mean_hand_foot_reach'] is not None else 0.0
    features['norm_max_hand_foot_reach'] = safe_divide(features['max_hand_foot_reach'], board_span_norm) if features['max_hand_foot_reach'] is not None else 0.0
    features['norm_min_hand_foot_reach'] = safe_divide(features['min_hand_foot_reach'], board_span_norm) if features['min_hand_foot_reach'] is not None else 0.0
    features['norm_std_hand_foot_reach'] = safe_divide(features['std_hand_foot_reach'], board_span_norm) if features['std_hand_foot_reach'] is not None else 0.0

    # Other features
    features['angle_x_holds'] = features['angle'] * features['num_holds']


    return features


# Testing
def main():
    difficulty_grades = adb.extract_difficulty_grades()

    # number of climbs and offset
    count = 20000
    offset = random.randint(0, adb.get_table_size("climbs") - count)

    # Fetch rows from the database
    # rows = adb.fetch_rows_from(count, offset)
    rows = adb.fetch_random_rows(count)

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
            record['edge_top'],
            )
        
        if features is None:
            continue

        record.update(features)
        climb_data.append(record)
        # Testing
        # print(f"Climb ID: {record['climb_id']}, Grade: {record['climb_grade']}, Features: {features}")

    df = pd.DataFrame(climb_data)

    # Remove rows with missing features
    df = df[df["climb_grade"].notna()]

    base_feature_columns = [
        "angle",
        "angle_squared",
        "is_nomatch", 
        "num_holds",
        "norm_height_gained",
        "hold_per_unit_height", 
        "norm_hold_density", 
        "norm_mean_hand_reach",
        "norm_max_hand_reach",
        "norm_mean_hand_foot_reach",
        "norm_min_hand_foot_reach",
        "norm_mean_y",
        ]

    feature_columns = [
        "angle", 
        "angle_squared", 
        "num_holds", 
        "angle_x_holds",
        "height_gained", 
        "hold_density", 
        "mean_hand_reach", 
        "max_hand_reach", 
        "std_hand_reach", 
        "hold_per_unit_height", 
        "is_nomatch", 
        "mean_hand_foot_reach", 
        "min_hand_foot_reach", 
        "std_hand_foot_reach",
        "hand_to_foot_density_ratio",
        ]
    
    normalized_feature_columns = [
        "angle", 
        "angle_squared", 
        "num_holds", 
        "angle_x_holds",
        "is_nomatch", 
        "norm_mean_y",
        "norm_height_gained",
        "norm_hold_density",
        "norm_mean_hand_reach",
        "norm_max_hand_reach",
        "norm_min_hand_reach",
        "norm_std_hand_reach",
        "norm_hand_spread_x",
        "norm_hand_spread_y",
        "norm_mean_hand_foot_reach",
        "norm_min_hand_foot_reach",
        "norm_std_hand_foot_reach"
    ]

    all_feature_columns = [
        "angle",
        "angle_squared",
        "num_holds",
        "num_hand_holds",
        "num_foot_holds",
        "start_holds",
        "finish_holds",
        "middle_holds",
        "is_nomatch",
        "mean_y",
        "std_y",
        "std_X",
        "min_y",
        "max_y",
        "min_x",
        "max_x",
        "range_y",
        "range_x",
        "start_height",
        "finish_height",
        "height_gained",
        "hold_per_unit_height",
        "board_area",
        "hold_density",
        "hand_hold_density",
        "foot_hold_density",
        "hand_to_foot_density_ratio",
        "mean_hand_reach",
        "max_hand_reach",
        "min_hand_reach",
        "std_hand_reach",
        "hand_spread_x",
        "hand_spread_y",
        "mean_hand_foot_reach",
        "max_hand_foot_reach",
        "min_hand_foot_reach",
        "std_hand_foot_reach",
        "norm_mean_y",
        "norm_std_y",
        "norm_std_x",
        "norm_min_y",
        "norm_max_y",
        "norm_min_x",
        "norm_max_x",
        "norm_range_y",
        "norm_range_x",
        "norm_start_height",
        "norm_finish_height",
        "norm_height_gained",
        "norm_board_area",
        "norm_hold_density",
        "norm_hand_hold_density",
        "norm_foot_hold_density",
        "norm_mean_hand_reach",
        "norm_max_hand_reach",
        "norm_min_hand_reach",
        "norm_std_hand_reach",
        "norm_hand_spread_x",
        "norm_hand_spread_y",
        "norm_mean_hand_foot_reach",
        "norm_max_hand_foot_reach",
        "norm_min_hand_foot_reach",
        "norm_std_hand_foot_reach",
        "angle_x_holds",
    ]

    # Fill NaN values in features with 0
    df[feature_columns] = df[feature_columns].fillna(0)
    df[base_feature_columns] = df[base_feature_columns].fillna(0)
    df[normalized_feature_columns] = df[normalized_feature_columns].fillna(0)
    df[all_feature_columns] = df[all_feature_columns].fillna(0)

    # Run models
    # lm.evaluate_all_grades_logistic(df, feature_columns)
    lm.evaluate_two_bucket_logistic(df, feature_columns)
    lm.evaluate_two_bucket_logistic(df, base_feature_columns)
    lm.evaluate_two_bucket_logistic(df, normalized_feature_columns)
    lm.evaluate_two_bucket_logistic(df, all_feature_columns)
    # lm.evaluate_bucketed_random_forest(df, feature_columns)


if __name__ == "__main__":
    main()
