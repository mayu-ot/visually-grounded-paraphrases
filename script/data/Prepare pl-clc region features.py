import sys
sys.path.append('./script/data/')
from converter import parse_entity_region_score

parse_entity_region_score('data/pl-clc/entity_region_scores.testSplit.txt',
                         'data/phrase_pair_test.csv',
                         'data/pl-clc/entity_region_scores_test.csv')

