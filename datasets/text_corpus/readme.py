variants = [
    'easy_raw_k1',
    'easy_raw_k3',
    'hard_raw_k1',
    'hard_raw_k3',
    'hard_cot_k1',
    'hard_cot_k3',
  ]

import pickle
# taking animalpose as an example
animalpose = pickle.load(open("interpolated_animalpose.pkl", 'rb'))
# getting the variant 'easy_raw_k1'
easy_k1_animalpose = animalpose['easy_raw_k1']
# getting an interpolated answer based on a (start,end) tuple
_ans = easy_k1_animalpose[(4, 2)]
