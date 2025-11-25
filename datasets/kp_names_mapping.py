ANIMAL_POSE = [
    'left eye',
    'right eye',
    'left ear',
    'right ear',
    'nose',
    'throat',
    'withers',
    'tail',
    'left-front leg',
    'right-front leg',
    'left-back leg',
    'right-back leg',
    'left-front knee',
    'right-front knee',
    'left-back knee',
    'right-back knee',
    'left-front paw',
    'right-front paw',
    'left-back paw',
    'right-back paw'
]
AWA = [
    # '_background_',
    'nose',
    'upper_jaw',
    'lower_jaw',
    'mouth_end_right',
    'mouth_end_left',
    'right_eye',
    'right_earbase',
    'right_earend',
    'right_antler_base',
    'right_antler_end',
    'left_eye',
    'left_earbase',
    'left_earend',
    'left_antler_base',
    'left_antler_end',
    'neck_base',
    'neck_end',
    'throat_base',
    'throat_end',
    'back_base',
    'back_end',
    'back_middle',
    'tail_base',
    'tail_end',
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
CUB = [
    'back',
    'beak',
    'belly',
    'breast',
    'crown',
    'forehead',
    'left eye',
    'left leg',
    'left wing',
    'nape',
    'right eye',
    'right leg',
    'right wing',
    'tail',
    'throat',
]
NABIRD = [
    'bill',  # namely beak
    'crown',
    'nape',
    'left eye',
    'right eye',
    'belly',
    'breast',
    'back',
    'tail',
    'left wing',
    'right wing',
]
DEEPFASHION2 = [

]

def get_mapped_kps_names(dataset_type, origin_kp_types, ORIGIN_FULL_KEYPOINT_TYPES):
    # assert dataset_type in ['ANIMAL_POSE', 'CUB', 'NABIRD', 'AWA', 'DEEPFASHION2']
    mapped_keypoints = eval(dataset_type)
    kps_names = []
    for each_kp_type in origin_kp_types:
        id = ORIGIN_FULL_KEYPOINT_TYPES.index(each_kp_type)
        kps_names.append(mapped_keypoints[id])

    return kps_names

def get_original_kps_names(dataset_type, mapped_kp_types, ORIGIN_FULL_KEYPOINT_TYPES):
    # assert dataset_type in ['ANIMAL_POSE', 'CUB', 'NABIRD', 'AWA', 'DEEPFASHION2']
    mapped_keypoints = eval(dataset_type)
    kps_names = []
    for each_kp_type in mapped_kp_types:
        id = mapped_keypoints.index(each_kp_type)
        kps_names.append(ORIGIN_FULL_KEYPOINT_TYPES[id])

    return kps_names
