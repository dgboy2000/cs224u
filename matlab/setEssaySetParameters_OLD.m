% order of final classifiers:

% allBestKappas = [maxKappaBoost,maxKappaLinReg,maxKappaEns,maxKappaSVM, maxKappaNN]
%                       1             2               3           4           5
% output/features.set1.dom1winner: Ensemble 0.8349 
% output/features.set2.dom1winner: boost 0.69209
% output/features.set2.dom2winner: linReg 0.68999
% output/features.set3.dom1winner: Ensemble 0.72128
% output/features.set4.dom1winner: linReg 0.77929
% output/features.set5.dom1winner: linReg 0.81144
% output/features.set6.dom1winner: linReg 0.80923
% output/features.set7.dom1winner: linReg 0.81815
% output/features.set8.dom1winner: Ensemble 0.71998


switch prefix2
    case 'output/ds.set1.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [5 ];

        allSVMReg = 2;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [2.1];
        
%         nnCV_AllHid = 1;
%         nnCV_regC = .3;
%         nnCV_regC_Wreg = 0;
        
        finalRegressionMethod = 3;
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
