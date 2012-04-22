% order of final classifiers:

% allBestKappas = [maxKappaBoost,maxKappaLinReg,maxKappaEns,maxKappaSVM, maxKappaNN]
%                       1             2               3           4           5

% output/features.set1.dom1       neural  0.82091 1       0.01    0.3
% output/features.set1.dom1       boost   0.81825 150     32      5
% output/features.set1.dom1       svm     0.82525 2
% output/features.set1.dom1       linReg  0.82171 welsch  2.1
% output/features.set1.dom1       ens     0.82643
% output/features.set1.dom1 winner: Ensemble 0.82643
% output/features.set2.dom1       neural  0.66517 1       0.01    0.3
% output/features.set2.dom1       boost   0.69574 2       110     20
% output/features.set2.dom1       svm     0.6683  0.5
% output/features.set2.dom1       linReg  0.68123 talwar  0.1
% output/features.set2.dom1       ens     0.68323
% output/features.set2.dom1 winner: boost 0.69574
% output/features.set2.dom2       neural  0.67652 1       0.01    0.3
% output/features.set2.dom2       boost   0.64445 60      150     25
% output/features.set2.dom2       svm     0.66971 0.5
% output/features.set2.dom2       linReg  0.68999 talwar  2.1
% output/features.set2.dom2       ens     0.67861
% output/features.set2.dom2 winner: linReg 0.68999
% output/features.set3.dom1       neural  0.69943 1       0.01    0.3
% output/features.set3.dom1       boost   0.69721 160     130     70
% output/features.set3.dom1       svm     0.67942 0.1
% output/features.set3.dom1       linReg  0.71483 welsch  1.339
% output/features.set3.dom1       ens     0.72037
% output/features.set3.dom1 winner: Ensemble 0.72037
% output/features.set4.dom1       neural  0.77028 1       0.01    0.3
% output/features.set4.dom1       boost   0.70893 70      110     3
% output/features.set4.dom1       svm     0.7555  0.1
% output/features.set4.dom1       linReg  0.77929 talwar  2.1
% output/features.set4.dom1       ens     0.75788
% output/features.set4.dom1 winner: linReg 0.77929
% output/features.set5.dom1       neural  0.80641 1       0.01    0.3
% output/features.set5.dom1       boost   0.80168 6       170     3
% output/features.set5.dom1       svm     0.80388 0.5
% output/features.set5.dom1       linReg  0.81144 bisquare        4
% output/features.set5.dom1       ens     0.80785
% output/features.set5.dom1 winner: linReg 0.81144
% output/features.set6.dom1       neural  0.80765 1       0.01    0.3
% output/features.set6.dom1       boost   0.77438 70      150     25
% output/features.set6.dom1       svm     0.78921 2
% output/features.set6.dom1       linReg  0.80923 talwar  2.1
% output/features.set6.dom1       ens     0.79787
% output/features.set6.dom1 winner: linReg 0.80923
% output/features.set7.dom1       neural  0.81446 1       0.01    0.3
% output/features.set7.dom1       boost   0.78714 64      150     3
% output/features.set7.dom1       svm     0.80542 1
% output/features.set7.dom1       linReg  0.81815 huber   0.1
% output/features.set7.dom1       ens     0.81453
% output/features.set7.dom1 winner: linReg 0.81815
% output/features.set8.dom1       neural  0.52414 1       0.01    0.3
% output/features.set8.dom1       boost   0.69782 200     170     25
% output/features.set8.dom1       svm     0.69885 0.5
% output/features.set8.dom1       linReg  0.69474 huber   4
% output/features.set8.dom1       ens     0.71765
% output/features.set8.dom1 winner: Ensemble 0.71765



switch prefix2
    case 'output/ds.set1.dom1',
        allTreeCV = [32 ];
        allMtryCV = [150 ];
        allNodeSizeCV = [5 ];

        allSVMReg = 2;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [2.1];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .3;
%         nnCV_regC_Wreg = 0;
        
        finalRegressionMethod = 1;
    case 'output/ds.set2.dom1',
        allTreeCV = [110 ];
        allMtryCV = [2 ];
        allNodeSizeCV = [20 ];
        
        allSVMReg = .5;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [0.1];
        
%         nnCV_AllHid = 2;
%         nnCV_regC = .3;
%         nnCV_regC_Wreg = 0.6;
        
        finalRegressionMethod = 1;
    case 'output/ds.set2.dom2',
        allTreeCV = [150 ];
        allMtryCV = [60 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = .5;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .2;
%         nnCV_regC_Wreg = 0.3;

        finalRegressionMethod = 2;
    case 'output/ds.set3.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [70 ];
        
        allSVMReg = .1;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [1.339];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .3;
%         nnCV_regC_Wreg = 0.1;
        
        finalRegressionMethod = 3;
    case 'output/ds.set4.dom1',
        allTreeCV = [110 ];
        allMtryCV = [70 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = .1;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
%         nnCV_AllHid = 2;
%         nnCV_regC = .5;
%         nnCV_regC_Wreg = 0.1;
 
        finalRegressionMethod = 2;
    case 'output/ds.set5.dom1',
        allTreeCV = [170 ];
        allMtryCV = [6 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = .5;
        
        wFcts = {'bisquare'}; %'fair','ols'
        tuningConst = [4];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .3;
%         nnCV_regC_Wreg = 0.4;
        
        finalRegressionMethod = 2;
    case 'output/ds.set6.dom1',
        allTreeCV = [150 ];
        allMtryCV = [70 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = 2;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
%         nnCV_AllHid = 2;
%         nnCV_regC = .6;
%         nnCV_regC_Wreg = 0.4;
        
        finalRegressionMethod = 2;
    case 'output/ds.set7.dom1',
        allTreeCV = [150 ];
        allMtryCV = [64 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = 1;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [0.1];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .5;
%         nnCV_regC_Wreg = 0.2;
        
        finalRegressionMethod = 2;
    case 'output/ds.set8.dom1',
        allTreeCV = [170 ];
        allMtryCV = [200 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = .5;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [4];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .4;
%         nnCV_regC_Wreg = 0.1;
        
        finalRegressionMethod = 3;
    otherwise
        error('different prefix2 than what is implemented?')
end

nnCV_AllHid = [1];
nnCV_regC = [0.01]
nnCV_regC_Wreg = [0.3]
