import re
import sqlite3
import pandas as pd
from pathlib import Path

import sklearn
from sklearn.linear_model import LinearRegression

from scripts.fetch_coords import extract_holes


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
placement_coords = extract_holes()

# Pawel defulted middle, unknown for now to catch unexpected hold ids
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
        coridnate = placement_coords.get(p_id, (None, None))
        hold_type = get_hold_type(h_id)
        is_hand_hold = h_id in HAND_HOLD_IDS
        is_foot_hold = h_id in FOOT_HOLD_IDS
        hold_data.append({
            'hold_id': p_id,
            'x': coridnate[0],
            'y': coridnate[1],
            'hold_type': hold_type,
            'is_hand_hold': is_hand_hold,
            'is_foot_hold': is_foot_hold
        })
    
    # testing
    print_hold_data(hold_data)

    if not hold_data:
        raise ValueError("No valid holds found in the frame.")

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    features = {}

    #implemented features for testing:

    # Base features
    features['angle'] = angle
    # features["angle_squared"] = float(angle) ** 2
    features['num_holds'] = len(holds)
    features['num_hand_holds'] = len([hold for hold in hold_data if hold['is_hand_hold']])
    features['num_foot_holds'] = len([hold for hold in hold_data if hold['is_foot_hold']])
    features['start_holds'] = len([hold for hold in hold_data if hold['hold_type'] == 'start'])
    features['finish_holds'] = len([hold for hold in hold_data if hold['hold_type'] == 'finish'])
    features['middle_holds'] = len([hold for hold in hold_data if hold['hold_type'] == 'middle'])

    # Spatial features
    features['mean_y'] = sum(hold['y'] for hold in hold_data) / len(hold_data)
    features['std_y'] = (sum((hold['y'] - center_y) ** 2 for hold in hold_data) / len(hold_data)) ** 0.5
    features['min_y'] = min(hold['y'] for hold in hold_data)
    features['max_y'] = max(hold['y'] for hold in hold_data)
    features['range_y'] = features['max_y'] - features['min_y']
    features['range_x'] = max(hold['x'] for hold in hold_data) - min(hold['x'] for hold in hold_data)
    features["height_gained"] = features["max_y"] - features["min_y"]
    
    features['start_height'] = min(hold['y'] for hold in hold_data if hold['hold_type'] == 'start') if features['start_holds'] > 0 else None
    features['finish_height'] = max(hold['y'] for hold in hold_data if hold['hold_type'] == 'finish') if features['finish_holds'] > 0 else None
    features['height_gained_from_start_to_finish'] = (features['finish_height'] - features['start_height']) if features['start_height'] is not None and features['finish_height'] is not None else None

    # Density features
    features['box_area'] = features['range_x'] * features['range_y'] if features['range_x'] and features['range_y'] else 1
    features['hold_density'] = features['num_holds'] / features['box_area'] if features['box_area'] else 0
    features['hand_hold_density'] = features['num_hand_holds'] / features['box_area'] if features['box_area'] else 0
    features['foot_hold_density'] = features['num_foot_holds'] / features['box_area'] if features['box_area'] else 0
    # features['hold_per_unit_height'] = features['num_holds'] / features['height_gained'] if features['height_gained'] else 0

    # Hand reach features
    if features['num_hand_holds'] >= 2:
        features['mean_hand_reach'] = sum(abs(hold_data[i]['x'] - hold_data[i-1]['x']) for i in range(1, len(hold_data)) if hold_data[i]['is_hand_hold'] and hold_data[i-1]['is_hand_hold']) / (features['num_hand_holds'] - 1)
        features['max_hand_reach'] = float(max(abs(hold_data[i]['x'] - hold_data[i-1]['x']) for i in range(1, len(hold_data)) if hold_data[i]['is_hand_hold'] and hold_data[i-1]['is_hand_hold']))
        features['std_hand_reach'] = (sum((abs(hold_data[i]['x'] - hold_data[i-1]['x']) - features['mean_hand_reach']) ** 2 for i in range(1, len(hold_data)) if hold_data[i]['is_hand_hold'] and hold_data[i-1]['is_hand_hold']) / (features['num_hand_holds'] - 1)) ** 0.5
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
    angle = 30  # Example angle
    frame = "p1r3p14r2p67r1p73r1p80r2p279r4"  # Example frame string
    features = extract_features_from_frame(angle, frame)
    # Print the extracted features
    for feature, value in features.items():
        print(f"{feature}: {value}")

if __name__ == "__main__":
    main()
