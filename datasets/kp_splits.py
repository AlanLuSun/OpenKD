def train_test_kp_set(dataset_type):
    if dataset_type == 'ANIMAL_POSE':
        training_kp_category_set = [
            # 'left eye',
            # 'right eye',
            'left ear',
            'right ear',
            'nose',
            # 'throat',
            # 'withers',
            # 'tail',
            'left-front leg',
            'right-front leg',
            'left-back leg',
            'right-back leg',
            # 'left-front knee',
            # 'right-front knee',
            # 'left-back knee',
            # 'right-back knee',
            'left-front paw',
            'right-front paw',
            'left-back paw',
            'right-back paw'
        ]

        testing_kp_category_set = [
            'left eye',
            'right eye',
            # 'left ear',
            # 'right ear',
            # 'nose',
            # 'throat',
            # 'withers',
            # 'tail',
            # 'left-front leg',
            # 'right-front leg',
            # 'left-back leg',
            # 'right-back leg',
            'left-front knee',
            'right-front knee',
            'left-back knee',
            'right-back knee',
            # 'left-front paw',
            # 'right-front paw',
            # 'left-back paw',
            # 'right-back paw'
        ]

        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2= 3  # 3 or 2 or 4 or above
        least_q_kp_num2= 3  # 3 or 2 or 4 or above
    elif dataset_type == 'AWA':
        training_kp_category_set = [
            'nose',
            # 'upper_jaw',
            # 'lower_jaw',
            # 'mouth_end_right',
            # 'mouth_end_left',
            # 'right_eye',
            'right_earbase',
            # 'right_earend',
            # 'right_antler_base',
            # 'right_antler_end',
            # 'left_eye',
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
            # 'front_left_knee',
            'front_left_paw',
            'front_right_thai',
            'front_right_paw',
            # 'front_right_knee',
            # 'back_left_knee',
            'back_left_paw',
            'back_left_thai',
            'back_right_thai',
            'back_right_paw',
            # 'back_right_knee',
            'belly_bottom',
            'body_middle_right',
            'body_middle_left',
        ]

        testing_kp_category_set = [
            # 'nose',
            # 'upper_jaw',
            # 'lower_jaw',
            # 'mouth_end_right',
            # 'mouth_end_left',
            'right_eye',
            # 'right_earbase',
            # 'right_earend',
            # # 'right_antler_base',
            # # 'right_antler_end',
            'left_eye',
            # 'left_earbase',
            # 'left_earend',
            # # 'left_antler_base',
            # # 'left_antler_end',
            # 'neck_base',
            # 'neck_end',
            # 'throat_base',
            # 'throat_end',
            # 'back_base',
            # 'back_end',
            # 'back_middle',
            # 'tail_base',
            # # 'tail_end',
            # 'front_left_thai',
            'front_left_knee',
            # 'front_left_paw',
            # 'front_right_thai',
            # 'front_right_paw',
            'front_right_knee',
            'back_left_knee',
            # 'back_left_paw',
            # 'back_left_thai',
            # 'back_right_thai',
            # 'back_right_paw',
            'back_right_knee',
            # 'belly_bottom',
            # 'body_middle_right',
            # 'body_middle_left',
        ]
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 3  # 3 or 2 or 4 or above
        least_q_kp_num2 = 3  # 3 or 2 or 4 or above
    elif dataset_type == 'CUB':
        training_kp_category_set = [
            'back',
            'beak',
            'belly',
            'breast',
            'crown',
            # 'forehead',
            # 'left eye',
            'left leg',
            # 'left wing',
            'nape',
            # 'right eye',
            'right leg',
            # 'right wing',
            'tail',
            'throat',
        ]
        testing_kp_category_set = [
            # 'back',
            # 'beak',
            # 'belly',
            # 'breast',
            # 'crown',
            'forehead',
            'left eye',
            # 'left leg',
            'left wing',
            # 'nape',
            'right eye',
            # 'right leg',
            'right wing',
            # 'tail',
            # 'throat',
        ]
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 3  # 3 or 2 or 4 or above
        least_q_kp_num2 = 3  # 3 or 2 or 4 or above
    elif dataset_type == 'NABIRD':
        training_kp_category_set = [
            'bill',  # namely beak
            'crown',
            'nape',
            # 'left eye',
            # 'right eye',
            'belly',
            'breast',
            'back',
            'tail',
            # 'left wing',
            # 'right wing',
        ]

        testing_kp_category_set = [
            # 'bill',  # namely beak
            # 'crown',
            # 'nape',
            'left eye',
            'right eye',
            # 'belly',
            # 'breast',
            # 'back',
            # 'tail',
            'left wing',
            'right wing',
        ]
        # least kp num for base keypoints
        least_s_kp_num = 3  # 3 or 2 or 4 or above
        least_q_kp_num = 3  # 3 or 2 or 4 or above
        # least kp num for novel keypoints
        least_s_kp_num2 = 2  # 3 or 2 or 4 or above
        least_q_kp_num2 = 2  # 3 or 2 or 4 or above
    elif dataset_type == 'DEEPFASHION2':
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

def get_full_kp_set(dataset_type):
    outs = train_test_kp_set(dataset_type)
    train_kp_set = outs[0]
    test_kp_set = outs[1]
    full_kp_set = train_kp_set + test_kp_set
    return full_kp_set