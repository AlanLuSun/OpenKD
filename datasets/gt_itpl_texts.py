def groundtruth_itpl_texts(dataset_type):
    if dataset_type == 'ANIMAL_POSE':
        gt_texts = [
            'left eye',
            'right eye',
            'left-front knee',
            'right-front knee',
            'left-back knee',
            'right-back knee',
        ]
        corpus_file_name = 'interpolated_animalpose'
    elif dataset_type == 'CUB':
        # gt_texts = [
        #     'forehead,
        #     'left eye or right eye',  # left eye or right eye, both are GT.
        #     'back, left wing, or right wing',  # back, left wing, or right wing
        #     'left wing or right wing',  # left wing, or right wing
        #     'left wing or right wing',  # left wing, or right wing
        #     'left wing or right wing',  # left wing, or right wing
        # ]
        gt_texts = [
            'forehead',
            'eye, left eye, right eye',  # left eye or right eye, both are GT.
            'back, wing, left wing, right wing',  # back, left wing, or right wing
            'wing, left wing, right wing',  # left wing, or right wing
            'wing, left wing, right wing',  # left wing, or right wing
            'wing, left wing, right wing',  # left wing, or right wing
        ]
        corpus_file_name = 'interpolated_cub'
    elif dataset_type == 'NABIRD':
        # gt_texts = [
        #     'forehead,
        #     'left eye or right eye',  # left eye or right eye, both are GT.
        #     'back, left wing, or right wing',  # back, left wing, or right wing
        #     'left wing or right wing',  # left wing, or right wing
        #     'left wing or right wing',  # left wing, or right wing
        #     'left wing or right wing',  # left wing, or right wing
        # ]
        gt_texts = [
            'forehead',
            'eye, left eye, right eye',  # left eye or right eye, both are GT.
            'back, wing, left wing, right wing',  # back, left wing, or right wing
            'wing, left wing, right wing',  # left wing, or right wing
            'wing, left wing, right wing',  # left wing, or right wing
            'wing, left wing, right wing',  # left wing, or right wing
        ]
        corpus_file_name = 'interpolated_nabird'
    elif dataset_type == 'AWA':
        gt_texts = [
            'left eye',
            'right eye',
            'left-front knee',
            'right-front knee',
            'left-back knee',
            'right-back knee',
        ]
        corpus_file_name = 'interpolated_awa'
    else:
        raise NotImplementedError
    
    return gt_texts, corpus_file_name