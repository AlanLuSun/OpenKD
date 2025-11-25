def train_test_kp_set(cfg):
    if cfg.DATASET.TYPE == 'ANIMAL_POSE':
        # classes = ['cat', 'cow', 'horse', 'sheep', 'dog']
        training_kp_category_set = [
            'l_eye',
            'r_eye',
            'l_ear',
            'r_ear',
            'nose',
            'throat',
            'withers',
            'tail',
            'l_f_leg',
            'r_f_leg',
            'l_b_leg',
            'r_b_leg',
            'l_f_knee',
            'r_f_knee',
            'l_b_knee',
            'r_b_knee',
            'l_f_paw',
            'r_f_paw',
            'l_b_paw',
            'r_b_paw'
        ]

        testing_kp_category_set = [
            'l_eye',
            'r_eye',
            'l_ear',
            'r_ear',
            'nose',
            'throat',
            'withers',
            'tail',
            'l_f_leg',
            'r_f_leg',
            'l_b_leg',
            'r_b_leg',
            'l_f_knee',
            'r_f_knee',
            'l_b_knee',
            'r_b_knee',
            'l_f_paw',
            'r_f_paw',
            'l_b_paw',
            'r_b_paw'
        ]

        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2= 3  # 3 or 2 or 4 or above
        least_q_kp_num2= 3  # 3 or 2 or 4 or above
    elif cfg.DATASET.TYPE == 'AWA':
        training_kp_category_set = [
            'nose',
            # 'upper_jaw',
            # 'lower_jaw',
            # 'mouth_end_right',
            # 'mouth_end_left',
            'right_eye',
            'right_earbase',
            # 'right_earend',
            # 'right_antler_base',
            # 'right_antler_end',
            'left_eye',
            'left_earbase',
            # 'left_earend',
            # 'left_antler_base',
            # 'left_antler_end',
            'neck_base',
            'neck_end',
            'throat_base',
            'throat_end',
            # 'back_base',
            # 'back_end',
            # 'back_middle',
            'tail_base',
            # 'tail_end',
            'front_left_thai',
            'front_left_knee',
            'front_left_paw',
            'front_right_thai',
            'front_right_paw',
            'front_right_knee',
            'back_left_knee',
            'back_left_paw',
            'back_left_thai',
            'back_right_thai',
            'back_right_paw',
            'back_right_knee',
            'belly_bottom',
            'body_middle_right',
            'body_middle_left',
        ]

        testing_kp_category_set = [
            'nose',
            # 'upper_jaw',
            # 'lower_jaw',
            # 'mouth_end_right',
            # 'mouth_end_left',
            'right_eye',
            'right_earbase',
            # 'right_earend',
            # 'right_antler_base',
            # 'right_antler_end',
            'left_eye',
            'left_earbase',
            # 'left_earend',
            # 'left_antler_base',
            # 'left_antler_end',
            'neck_base',
            'neck_end',
            'throat_base',
            'throat_end',
            # 'back_base',
            # 'back_end',
            # 'back_middle',
            'tail_base',
            # 'tail_end',
            'front_left_thai',
            'front_left_knee',
            'front_left_paw',
            'front_right_thai',
            'front_right_paw',
            'front_right_knee',
            'back_left_knee',
            'back_left_paw',
            'back_left_thai',
            'back_right_thai',
            'back_right_paw',
            'back_right_knee',
            'belly_bottom',
            'body_middle_right',
            'body_middle_left',
        ]
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 3  # 3 or 2 or 4 or above
        least_q_kp_num2 = 3  # 3 or 2 or 4 or above
    elif cfg.DATASET.TYPE == 'CUB':
        training_kp_category_set = [
            'back',
            'beak',
            'belly',
            'breast',
            'crown',
            'forehead',
            'left_eye',
            'left_leg',
            'left_wing',
            'nape',
            'right_eye',
            'right_leg',
            'right_wing',
            'tail',
            'throat',
        ]
        testing_kp_category_set = [
            'back',
            'beak',
            'belly',
            'breast',
            'crown',
            'forehead',
            'left_eye',
            'left_leg',
            'left_wing',
            'nape',
            'right_eye',
            'right_leg',
            'right_wing',
            'tail',
            'throat',
        ]
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 3  # 3 or 2 or 4 or above
        least_q_kp_num2 = 3  # 3 or 2 or 4 or above
    elif cfg.DATASET.TYPE == 'NABIRD':
        training_kp_category_set = [
            'bill',  # namely beak
            'crown',
            'nape',
            'left_eye',
            'right_eye',
            'belly',
            'breast',
            'back',
            'tail',
            'left_wing',
            'right_wing',
        ]

        testing_kp_category_set = [
            'bill',  # namely beak
            'crown',
            'nape',
            'left_eye',
            'right_eye',
            'belly',
            'breast',
            'back',
            'tail',
            'left_wing',
            'right_wing',
        ]
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 2  # 3 or 2 or 4 or above
        least_q_kp_num2 = 2  # 3 or 2 or 4 or above
    elif cfg.DATASET.TYPE == 'DEEPFASHION2':
        '''
        For DeepFashion2 dataset, 6 upper-body clothing categories are set as seen categories and 7 lower-body
        clothing categories are set as unseen ones.
        '''
        training_kp_category_set = []
        testing_kp_category_set = []
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 3  # 3 or 2 or 4 or above
        least_q_kp_num2 = 3  # 3 or 2 or 4 or above
    else:
        raise NotImplementedError

    return training_kp_category_set, testing_kp_category_set, least_s_kp_num, least_q_kp_num, least_s_kp_num2, least_q_kp_num2