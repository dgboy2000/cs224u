import math
import re


def parse_features(line):
    parts = re.split("\\s+", line)
    grade = int(parts[0])
    features = []
    for feature in parts[2:]:
        if len(feature) == 0:
            break
        feat_num, feat_val = re.split(":", feature)
        features.append(feat_val)
    return grade, features

features = []
grades = []

f = open('/tmp/svm_rank_features.dat')
for line in f.readlines():
    if len(line) == 0:
        break
    grade, feat_vec = parse_features(line)
    features.append(feat_vec)
    grades.append(grade)
f.close

f = open('/tmp/cs224u_features.dat', 'w')
f.write("%d\t%d\n" %(len(features), len(features[0])))
for i in range(len(features)):
    f.write("%d\t%s\n" %(grades[i], "\t".join(features[i])))
f.close()