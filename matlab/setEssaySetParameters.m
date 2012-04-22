% order of final classifiers:



% allBestKappas = [maxKappaBoost,maxKappaLinReg,maxKappaEns,maxKappaSVM, maxKappaNN]
%                       1             2               3           4           5

% writeTextFile('_allResults.txt',{prefix 'boost' maxKappaBoost bestParams.mtry  bestParams.numTrees ...
%     bestParams.nodeSize},struct('separator', '\t')) %bestParams.corr_bias bestParams.importance always 1

nnCV_AllHid = [1];
nnCV_regC = [0.01]
nnCV_regC_Wreg = [0.3]


switch prefix2
    case 'output/ds.set1.dom1',
        allTreeCV = [110 ];
        allMtryCV = [200 ];
        allNodeSizeCV = [5 ];

        allSVMReg = 1;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [4];
        
        nnCV_AllHid = 1;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0.1;
        
        finalRegressionMethod = 1;
    case 'output/ds.set2.dom1',
        allTreeCV = [130 ];
        allMtryCV = [128 ];
        allNodeSizeCV = [30 ];
        
        allSVMReg = .1;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.9];
        
        nnCV_AllHid = 1;
        nnCV_regC = .2;
        nnCV_regC_Wreg = 0.6;
        
        finalRegressionMethod = 1;
    case 'output/ds.set2.dom2',
        allTreeCV = [150 ];
        allMtryCV = [60 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = 2;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [2.9];
        
        nnCV_AllHid = 1;
        nnCV_regC = .5;
        nnCV_regC_Wreg = 0.6;

        finalRegressionMethod = 2;
    case 'output/ds.set3.dom1',
        allTreeCV = [130 ];
        allMtryCV = [200 ];
        allNodeSizeCV = [70 ];
        
        allSVMReg = .1;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0.3;
        
        finalRegressionMethod = 2;
    case 'output/ds.set4.dom1',
        allTreeCV = [100 ];
        allMtryCV = [32 ];
        allNodeSizeCV = [5 ];
        
        allSVMReg = .1;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .1;
        nnCV_regC_Wreg = 0.6;

        finalRegressionMethod = 2;
    case 'output/ds.set5.dom1',
        allTreeCV = [110 ];
        allMtryCV = [64 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = .1;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [0.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0;
        
        finalRegressionMethod = 2;
    case 'output/ds.set6.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = 2;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = 0;
        nnCV_regC_Wreg = 0.6;
        
        finalRegressionMethod = 2;
    case 'output/ds.set7.dom1',
        allTreeCV = [200 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [5 ];
        
        allSVMReg = 2;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [0.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .6;
        nnCV_regC_Wreg = 0;
        
        finalRegressionMethod = 2;
    case 'output/ds.set8.dom1',
        allTreeCV = [100 ];
        allMtryCV = [64 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = .5;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [4];
        
        nnCV_AllHid = 1;
        nnCV_regC = .2;
        nnCV_regC_Wreg = 0;
        
        finalRegressionMethod = 3;
    otherwise
        error('different prefix2 than what is implemented?')
end

