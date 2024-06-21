import numpy as np
import pandas as pd
import math


def convert_from_NED_to_Robotic(angle):
    """
    convert radian angle based on NED frame to radian angle based on the robotics frame

    Args:
        angle (float): angle in NED frame in radians
    """

    if 0 <= angle <= np.pi * 1/2:
        converted_angle = np.pi * 1/2 - angle
        return converted_angle

    elif np.pi * 1/2 < angle < np.pi * 3/2:
        converted_angle = - angle + np.pi * 1/2
        return converted_angle

    else: # 270 <= angle < 360
        converted_angle = (np.pi * 5/2) - angle
        return converted_angle


def convert_rad_angle_from_robotic_to_NED(angle):
    """
    convert radian angle based on robotic frame to radian angle based on the NED frame

    Arguments: 
        - angle: robotic heading angle (0: right, +pi and -pi) in radian

    Returns:
        - transformed_angle (NED clockwise, 0 northup) in radians
    """

    # [000, 270) 1st, 4th, 3rd quadrant
    if -math.pi < angle <= math.pi * 1/2:
        transformed_angle = (math.pi * 1/2) - angle
        return transformed_angle
    # [270, 360) 2nd quadrant
    elif math.pi * 1/2 < angle <= math.pi:
        transformed_angle = (math.pi * 2) - angle + (math.pi * 1/2)
        return transformed_angle


def get_trajectory_before_pass(df_input, obj_id_array):
    """
    Args:
        - df_input: data_frame for training or test

    Returns:
        - cropped dataframe with position only ahead for predicting the direction
    """
    df_cropped = pd.DataFrame()
    for idx, obj_id in enumerate(obj_id_array):
        # TODO backside too
        df_cropped = pd.concat([df_cropped, df_input.loc[(df_input['obj_index'] == obj_id) & \
                                                         (df_input['x'] > 0.0)]], ignore_index=True)
    return df_cropped


def check_dimension(x_data, y_data):
    if len(x_data) == len(y_data):
        print("Input, output dimension is same. good to go")
    else:
        raise ValueError("Input, output dimension is different. Double check!")