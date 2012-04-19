% tic

allKappas = [];
prefix='output/features.set1.dom1';prefix2='output/ds.set1.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set2.dom1';prefix2='output/ds.set2.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set2.dom2';prefix2='output/ds.set2.dom2';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set3.dom1';prefix2='output/ds.set3.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set4.dom1';prefix2='output/ds.set4.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set5.dom1';prefix2='output/ds.set5.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set6.dom1';prefix2='output/ds.set6.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set7.dom1';prefix2='output/ds.set7.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]
prefix='output/features.set8.dom1';prefix2='output/ds.set8.dom1';addpath(genpath('matlab'));trainTestEssayPipe
allKappas = [allKappas winningKappa]


meanQuadraticWeightedKappa(allKappas)


% BEST others:
% set 1
% Kappa Score 0.866116
% Kappa Score 0.838473
% BOOST alone 0.84!
% --
% 
% Train/Test Scores: (ESSAY_SET #2, DOMAIN 1)
% Kappa Score 0.756253
% Kappa Score 0.699274
% BOOST alone 0.7095!
% --
% 
% Train/Test Scores: (ESSAY_SET #2, DOMAIN 2)
% Kappa Score 0.738223
% Kappa Score 0.686292
% RS: WORSE: 0.68381
% allBestKappas =    0.6384    0.6790    0.6717    0.6697 LinReg Wins with 0.67902
% --
% 
% Train/Test Scores: (ESSAY_SET #3, DOMAIN 1)
% Kappa Score 0.698073
% Kappa Score 0.712269
% BETTER (only buggy ensemble):  0.71259
% --
% 
% Train/Test Scores: (ESSAY_SET #4, DOMAIN 1)
% Kappa Score 0.829644
% Kappa Score 0.780711
% WORSE? only 0.7719 --------> what's different between the regressions?
% --
% 
% Train/Test Scores: (ESSAY_SET #5, DOMAIN 1)
% Kappa Score 0.831516
% Kappa Score 0.813827
% MHHH ONLY   0.80993
% --
% 
% Train/Test Scores: (ESSAY_SET #6, DOMAIN 1)
% Kappa Score 0.815425
% Kappa Score 0.809794
% MHHH ONLY   0.80612
% --
% 
% Train/Test Scores: (ESSAY_SET #7, DOMAIN 1)
% Kappa Score 0.806354
% Kappa Score 0.813625
% a bit worse 0.81305
% --
% 
% Train/Test Scores: (ESSAY_SET #8, DOMAIN 1)
% Kappa Score 0.805900
% Kappa Score 0.704160
% RS: linreg  0.70914 huber
% RS: BETTER(only buggy ensemble) 0.72397
% --
% 
% Overall Train / Test
% Kappa Score 0.799282
% Kappa Score 0.768033

% MY OVERALL: 
% Kappa Score 0.894346
% Kappa Score 0.768745