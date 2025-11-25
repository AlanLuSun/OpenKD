Dataset structure is as follows:

1. ANIMAL_POSE_test_prompts_1000_parse.json
+a dict
+--cat (list)
   +--(anno_id1, prompt_dict1)
   +--(anno_id2, prompt_dict2)
   .
   .
   .
   +--(anno_idn, prompt_dictn)
+--dog (list)
+--cow (list)
+--horse (list)
+--sheep (list)

Each prompt_dict has four key-value pairs like: 
{'prompt': 'pinpointing the right ear, right-front leg, left-back leg, right-back knee.', 
'category': 'cat', 
'keypoints': ['right ear', 'right-front leg', 'left-back leg', 'right-back knee'], 
'parse': ['n/a', ['right ear', 'right front leg', 'left back leg', 'right back knee']]}


2. AWA_test_prompts_1000_parse.json and others are a list directly contain 1000 text prompts as follows:
+a list
+--(anno_id1, prompt_dict1)
+--(anno_id2, prompt_dict2)
.
.
.
+--(anno_idn, prompt_dictn)
