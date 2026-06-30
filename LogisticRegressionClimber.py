import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

climbing_data = {
    "climb_1": {"hold_density": 0.26, "distance_ft": 21.2, "angle_deg": 46, "v_rating": 10},
    "climb_2": {"hold_density": 0.81, "distance_ft": 10.5, "angle_deg": 24, "v_rating": 2},
    "climb_3": {"hold_density": 0.82, "distance_ft": 6.0, "angle_deg": 7, "v_rating": 1},
    "climb_4": {"hold_density": 0.70, "distance_ft": 8.5, "angle_deg": 17, "v_rating": 3},
    "climb_5": {"hold_density": 0.40, "distance_ft": 19.6, "angle_deg": 49, "v_rating": 10},
    "climb_6": {"hold_density": 0.72, "distance_ft": 9.7, "angle_deg": 31, "v_rating": 4},
    "climb_7": {"hold_density": 0.28, "distance_ft": 20.2, "angle_deg": 49, "v_rating": 11},
    "climb_8": {"hold_density": 0.21, "distance_ft": 20.6, "angle_deg": 52, "v_rating": 12},
    "climb_9": {"hold_density": 0.62, "distance_ft": 15.0, "angle_deg": 34, "v_rating": 5},
    "climb_10": {"hold_density": 0.36, "distance_ft": 19.5, "angle_deg": 35, "v_rating": 8},
    "climb_11": {"hold_density": 0.73, "distance_ft": 12.8, "angle_deg": 32, "v_rating": 4},
    "climb_12": {"hold_density": 0.33, "distance_ft": 16.5, "angle_deg": 49, "v_rating": 9},
    "climb_13": {"hold_density": 0.20, "distance_ft": 20.5, "angle_deg": 54, "v_rating": 12},
    "climb_14": {"hold_density": 0.77, "distance_ft": 8.1, "angle_deg": 19, "v_rating": 1},
    "climb_15": {"hold_density": 0.72, "distance_ft": 8.1, "angle_deg": 14, "v_rating": 2},
    "climb_16": {"hold_density": 0.39, "distance_ft": 20.5, "angle_deg": 45, "v_rating": 10},
    "climb_17": {"hold_density": 0.24, "distance_ft": 21.1, "angle_deg": 50, "v_rating": 11},
    "climb_18": {"hold_density": 0.40, "distance_ft": 18.6, "angle_deg": 47, "v_rating": 10},
    "climb_19": {"hold_density": 0.32, "distance_ft": 21.2, "angle_deg": 51, "v_rating": 12},
    "climb_20": {"hold_density": 0.58, "distance_ft": 11.3, "angle_deg": 37, "v_rating": 5},
    "climb_21": {"hold_density": 0.49, "distance_ft": 17.8, "angle_deg": 48, "v_rating": 9},
    "climb_22": {"hold_density": 0.64, "distance_ft": 15.5, "angle_deg": 28, "v_rating": 6},
    "climb_23": {"hold_density": 0.69, "distance_ft": 9.9, "angle_deg": 14, "v_rating": 2},
    "climb_24": {"hold_density": 0.38, "distance_ft": 19.1, "angle_deg": 44, "v_rating": 9},
    "climb_25": {"hold_density": 0.75, "distance_ft": 7.6, "angle_deg": 11, "v_rating": 2},
    "climb_26": {"hold_density": 0.72, "distance_ft": 6.6, "angle_deg": 17, "v_rating": 1},
    "climb_27": {"hold_density": 0.30, "distance_ft": 18.1, "angle_deg": 54, "v_rating": 9},
    "climb_28": {"hold_density": 0.39, "distance_ft": 17.7, "angle_deg": 49, "v_rating": 8},
    "climb_29": {"hold_density": 0.40, "distance_ft": 20.9, "angle_deg": 51, "v_rating": 10},
    "climb_30": {"hold_density": 0.71, "distance_ft": 11.4, "angle_deg": 23, "v_rating": 4},
    "climb_31": {"hold_density": 0.74, "distance_ft": 11.9, "angle_deg": 24, "v_rating": 2},
    "climb_32": {"hold_density": 0.76, "distance_ft": 13.5, "angle_deg": 26, "v_rating": 4},
    "climb_33": {"hold_density": 0.89, "distance_ft": 7.3, "angle_deg": 16, "v_rating": 1},
    "climb_34": {"hold_density": 0.33, "distance_ft": 18.1, "angle_deg": 41, "v_rating": 9},
    "climb_35": {"hold_density": 0.35, "distance_ft": 24.7, "angle_deg": 60, "v_rating": 12},
    "climb_36": {"hold_density": 0.38, "distance_ft": 18.2, "angle_deg": 44, "v_rating": 7},
    "climb_37": {"hold_density": 0.20, "distance_ft": 20.4, "angle_deg": 64, "v_rating": 12},
    "climb_38": {"hold_density": 0.70, "distance_ft": 8.2, "angle_deg": 7, "v_rating": 1},
    "climb_39": {"hold_density": 0.26, "distance_ft": 20.7, "angle_deg": 61, "v_rating": 12},
    "climb_40": {"hold_density": 0.37, "distance_ft": 17.5, "angle_deg": 44, "v_rating": 8},
    "climb_41": {"hold_density": 0.80, "distance_ft": 12.2, "angle_deg": 25, "v_rating": 3},
    "climb_42": {"hold_density": 0.25, "distance_ft": 23.8, "angle_deg": 56, "v_rating": 11},
    "climb_43": {"hold_density": 0.60, "distance_ft": 15.9, "angle_deg": 34, "v_rating": 7},
    "climb_44": {"hold_density": 0.76, "distance_ft": 8.7, "angle_deg": 10, "v_rating": 1},
    "climb_45": {"hold_density": 0.58, "distance_ft": 11.9, "angle_deg": 15, "v_rating": 3},
    "climb_46": {"hold_density": 0.90, "distance_ft": 10.1, "angle_deg": 7, "v_rating": 1},
    "climb_47": {"hold_density": 0.65, "distance_ft": 10.8, "angle_deg": 23, "v_rating": 3},
    "climb_48": {"hold_density": 0.40, "distance_ft": 21.7, "angle_deg": 54, "v_rating": 11},
    "climb_49": {"hold_density": 0.24, "distance_ft": 22.1, "angle_deg": 52, "v_rating": 12},
    "climb_50": {"hold_density": 0.33, "distance_ft": 19.6, "angle_deg": 49, "v_rating": 10},
    "climb_51": {"hold_density": 0.19, "distance_ft": 22.1, "angle_deg": 56, "v_rating": 11},
    "climb_52": {"hold_density": 0.83, "distance_ft": 6.2, "angle_deg": 16, "v_rating": 0},
    "climb_53": {"hold_density": 0.63, "distance_ft": 11.1, "angle_deg": 16, "v_rating": 3},
    "climb_54": {"hold_density": 0.70, "distance_ft": 8.3, "angle_deg": 25, "v_rating": 2},
    "climb_55": {"hold_density": 0.57, "distance_ft": 17.9, "angle_deg": 32, "v_rating": 7},
    "climb_56": {"hold_density": 0.47, "distance_ft": 21.7, "angle_deg": 58, "v_rating": 10},
    "climb_57": {"hold_density": 0.33, "distance_ft": 20.9, "angle_deg": 58, "v_rating": 12},
    "climb_58": {"hold_density": 0.79, "distance_ft": 12.9, "angle_deg": 17, "v_rating": 3},
    "climb_59": {"hold_density": 0.95, "distance_ft": 5.8, "angle_deg": 15, "v_rating": 0},
    "climb_60": {"hold_density": 0.45, "distance_ft": 17.1, "angle_deg": 42, "v_rating": 7},
    "climb_61": {"hold_density": 0.26, "distance_ft": 23.7, "angle_deg": 52, "v_rating": 12},
    "climb_62": {"hold_density": 0.58, "distance_ft": 10.0, "angle_deg": 30, "v_rating": 4},
    "climb_63": {"hold_density": 0.91, "distance_ft": 4.8, "angle_deg": 11, "v_rating": 0},
    "climb_64": {"hold_density": 0.55, "distance_ft": 17.6, "angle_deg": 35, "v_rating": 8},
    "climb_65": {"hold_density": 0.35, "distance_ft": 15.8, "angle_deg": 44, "v_rating": 8},
    "climb_66": {"hold_density": 0.44, "distance_ft": 19.5, "angle_deg": 57, "v_rating": 10},
    "climb_67": {"hold_density": 0.34, "distance_ft": 19.2, "angle_deg": 48, "v_rating": 9},
    "climb_68": {"hold_density": 0.59, "distance_ft": 15.1, "angle_deg": 31, "v_rating": 6},
    "climb_69": {"hold_density": 0.58, "distance_ft": 13.3, "angle_deg": 22, "v_rating": 4},
    "climb_70": {"hold_density": 0.46, "distance_ft": 15.5, "angle_deg": 33, "v_rating": 6},
    "climb_71": {"hold_density": 0.35, "distance_ft": 20.1, "angle_deg": 60, "v_rating": 12},
    "climb_72": {"hold_density": 0.52, "distance_ft": 16.6, "angle_deg": 41, "v_rating": 9},
    "climb_73": {"hold_density": 0.56, "distance_ft": 11.4, "angle_deg": 19, "v_rating": 4},
    "climb_74": {"hold_density": 0.67, "distance_ft": 9.2, "angle_deg": 27, "v_rating": 3},
    "climb_75": {"hold_density": 0.25, "distance_ft": 23.7, "angle_deg": 59, "v_rating": 11},
    "climb_76": {"hold_density": 0.33, "distance_ft": 19.0, "angle_deg": 39, "v_rating": 8},
    "climb_77": {"hold_density": 0.25, "distance_ft": 21.9, "angle_deg": 46, "v_rating": 10},
    "climb_78": {"hold_density": 0.86, "distance_ft": 6.6, "angle_deg": 11, "v_rating": 1},
    "climb_79": {"hold_density": 0.75, "distance_ft": 9.4, "angle_deg": 24, "v_rating": 3},
    "climb_80": {"hold_density": 0.65, "distance_ft": 11.0, "angle_deg": 33, "v_rating": 4},
    "climb_81": {"hold_density": 0.75, "distance_ft": 6.6, "angle_deg": 6, "v_rating": 0},
    "climb_82": {"hold_density": 0.81, "distance_ft": 5.2, "angle_deg": 18, "v_rating": 0},
    "climb_83": {"hold_density": 0.81, "distance_ft": 9.9, "angle_deg": 17, "v_rating": 2},
    "climb_84": {"hold_density": 0.76, "distance_ft": 9.2, "angle_deg": 13, "v_rating": 0},
    "climb_85": {"hold_density": 0.76, "distance_ft": 11.3, "angle_deg": 19, "v_rating": 2},
    "climb_86": {"hold_density": 0.73, "distance_ft": 7.3, "angle_deg": 16, "v_rating": 2},
    "climb_87": {"hold_density": 0.36, "distance_ft": 20.3, "angle_deg": 56, "v_rating": 12},
    "climb_88": {"hold_density": 0.29, "distance_ft": 18.0, "angle_deg": 54, "v_rating": 10},
    "climb_89": {"hold_density": 0.66, "distance_ft": 16.0, "angle_deg": 41, "v_rating": 6},
    "climb_90": {"hold_density": 0.79, "distance_ft": 13.3, "angle_deg": 27, "v_rating": 3},
    "climb_91": {"hold_density": 0.44, "distance_ft": 16.0, "angle_deg": 31, "v_rating": 6},
    "climb_92": {"hold_density": 0.62, "distance_ft": 16.6, "angle_deg": 39, "v_rating": 6},
    "climb_93": {"hold_density": 0.57, "distance_ft": 13.2, "angle_deg": 24, "v_rating": 4},
    "climb_94": {"hold_density": 0.94, "distance_ft": 5.6, "angle_deg": 15, "v_rating": 0},
    "climb_95": {"hold_density": 0.46, "distance_ft": 17.7, "angle_deg": 44, "v_rating": 7},
    "climb_96": {"hold_density": 0.59, "distance_ft": 9.4, "angle_deg": 19, "v_rating": 3},
    "climb_97": {"hold_density": 0.91, "distance_ft": 7.2, "angle_deg": 16, "v_rating": 1},
    "climb_98": {"hold_density": 0.59, "distance_ft": 16.5, "angle_deg": 31, "v_rating": 6},
    "climb_99": {"hold_density": 0.76, "distance_ft": 9.4, "angle_deg": 5, "v_rating": 0},
    "climb_100": {"hold_density": 0.54, "distance_ft": 12.7, "angle_deg": 24, "v_rating": 4},
}

df = pd.DataFrame.from_dict(climbing_data, orient="index")

X = df[["hold_density", "distance_ft", "angle_deg"]]
y = df["v_rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')