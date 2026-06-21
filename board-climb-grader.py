import re
import pandas as pd

# Alex King and Ben Hawkins
# board-climb-grader.py

## Board Constants
x_min = -24
x_max = 168
y_min = 0
y_max = 156
board_width = x_max - x_min
board_height = y_max - y_min

# Hold ID
HAND_HOLD_IDS = {12, 13, 14}
FOOT_HOLD_IDS = {15}

HOLD_MAPPING = {
    12: 'start',
    13: 'middle',
    14: 'finish',
    15: 'foot'
}

PLACEMENTS_PATH = 'placements.csv'

df_placements = pd.read_csv(PLACEMENTS_PATH)

# Copied need to be looked over
placement_coords = {
    int(row["placement_id"]): (row["x"], row["y"])
    for _, row in df_placements.iterrows()
}

# Pawel defulted middle, unknown for now to catch unexpected hold ids
def get_hold_type(hold_id):
    return HOLD_MAPPING.get(hold_id, 'unknown')

def parse_frame(frame):
    if isinstance(frame, str) or isinstance(frame.strip(), str):
        numbers = re.findall(r"p(\d+)r(\d+)", frame)
        return [(int(p), int(r)) for p, r in numbers]
    else:
        return None
    
def extract_features(angle, frame):
    holds = parse_frame(frame)
    if holds is None:
        raise ValueError("Invalid frame format.")
    
    hold_data = []
    for p, r in holds:
        coridnate = cordinate_set.get(p, (None, None))
        hold_type = get_hold_type(r)
        is_hand_hold = r in HAND_HOLD_IDS
        is_foot_hold = r in FOOT_HOLD_IDS
        hold_data.append({
            'hold_id': p,
            'x': coridnate[0],
            'y': coridnate[1],
            'hold_type': hold_type,
            'is_hand_hold': is_hand_hold,
            'is_foot_hold': is_foot_hold
        })

    if not hold_data:
        raise ValueError("No valid holds found in the frame.")

    features = {
        'angle': angle,
        'num_holds': len(holds),
        'num_hand_holds': sum(1 for p, r in holds if p in HAND_HOLD_IDS),
        'num_foot_holds': sum(1 for p, r in holds if p in FOOT_HOLD_IDS),
        'holds': hold_data

    }
    
    
    
    return features