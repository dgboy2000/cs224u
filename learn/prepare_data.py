import math
import os
import random
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

def write_dataset(filename, features, grades):
    f = open(filename, 'w')
    f.write("%d\t%d\n" %(len(features), len(features[0])))
    for i in range(len(features)):
        f.write("%d\t%s\n" %(grades[i], "\t".join([str(feat) for feat in features[i]])))
    f.close()


if __name__ == '__main__':
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

    write_dataset('/tmp/cs224u_features.dat', features, grades)


























