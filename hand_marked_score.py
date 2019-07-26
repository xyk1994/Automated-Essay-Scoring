import numpy as np
import pandas as pd

csv_input = pd.read_csv(filepath_or_buffer="essay_set1_with_feature.tsv", delimiter="\t", engine="python", error_bad_lines=False)
Score = csv_input['domain1_score']
score = np.array(Score)

max_score = 12
min_score = 2

hand_marked_score = (score - min_score) / (max_score - min_score)
print(hand_marked_score)
hand_marked_score = pd.DataFrame(hand_marked_score)
hand_marked_score.to_csv('data/hand_marked_score_set1.tsv', sep='\t')
