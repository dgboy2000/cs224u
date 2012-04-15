# Main running script

import glob
import os
import params
import Run
import sys
from score import KappaScore, MeanKappaScore

RUN_VAL = True
RUN_KAGGLE = False

if RUN_VAL:
    train_mean_kappa = MeanKappaScore()
    val_mean_kappa = MeanKappaScore()

    for essay_set in params.ESSAY_SETS:
        total_domains = 1
        if essay_set == 2:
            total_domains = 2

        for domain in range(1, total_domains+1):
            run = Run.Run()
            run.setup('data/c_train.utf8ignore.tsv', 'data/c_val.utf8ignore.tsv', essay_set, domain)
            run.extract()
            run.learn()
            run.predict()
            train_score, test_score = run.eval()
            run.dump()

            train_mean_kappa.add(train_score)
            val_mean_kappa.add(test_score)
            print "--\n"

    print "Overall Train / Test"
    print "Kappa Score %f" %train_mean_kappa.mean_quadratic_weighted_kappa()
    print "Kappa Score %f" %val_mean_kappa.mean_quadratic_weighted_kappa()

if RUN_KAGGLE:
    fd = open('data/kaggle_out.tsv', 'w')
    fd.write('prediction_id\tessay_id\tessay_set\tessay_weight\tpredicted_score\n')
    for essay_set in params.ESSAY_SETS:
        total_domains = 1
        if essay_set == 2:
            total_domains = 2

        for domain in range(1, total_domains+1):
            run = Run.Run()
            run.setup('data/training_set_rel3.utf8ignore.tsv', 'data/valid_set.utf8ignore.tsv', essay_set, domain)
            run.extract()
            run.learn()
            run.predict()

            run.outputKaggle(fd)

