
cvParams.useTextFeat = 0;

%prefix='output/features.set1.dom1';prefix2='output/ds.set1.dom1';
%matlab/trainTestEssayPipe
file_trn = [prefix2 '.train' '.matOut'];
file_tst = [prefix2 '.test' '.matOut'];
delete(file_trn)
delete(file_tst)

%compile everything
% if strcmpi(computer,'PCWIN') |strcmpi(computer,'PCWIN64')
%     cd matlab/randomforest-matlab/RF_Reg_C
%     compile_windows
%     cd ../../..
% else
%     cd matlab/randomforest-matlab/RF_Reg_C
%     compile_linux
%     cd ../../..
% end

[X_trn,Y_trn,Text_trn] = getFeatAndGrade(prefix,prefix2,'train');
[X_tst,Y_tst,Text_tst] = getFeatAndGrade(prefix,prefix2,'test');

if cvParams.useTextFeat
    if ~exist([prefix2 '.textFeat.mat'],'file')
        [X_trn_text,X_tst_text] = extractTextFeatures(Text_trn,Text_tst,prefix2);
        save([prefix2 '.textFeat.mat'],'X_trn_text','X_tst_text')
    else
        load([prefix2 '.textFeat.mat'],'X_trn_text','X_tst_text')
    end
    
    X_trn = [X_trn X_trn_text];
    X_tst = [X_tst X_tst_text];
end

%save('allDataSet2Dom2.mat','X_trn','Y_trn','Text_trn','X_tst','Y_tst','Text_tst','X_trn_text','X_tst_text')

%% normalize features
m = mean(X_trn,1);
X_trn = bsxfun(@minus,X_trn,m);
stdev = 1./std(X_trn,1);
stdev(isinf(stdev)) = 1;
X_trn = bsxfun(@times,stdev,X_trn);
%X_trn = bsxfun(@max,bsxfun(@min,X_trn,3*stdev),-3*stdev);
X_tst = bsxfun(@minus,X_tst,m);
X_tst = bsxfun(@times,stdev,X_tst);
%X_tst = bsxfun(@max,bsxfun(@min,X_tst,3*stdev),-3*stdev);



%% Set best parameters for each set here
% different CV sets for diff. essay sets
switch prefix2
    case 'output/ds.set1.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [5 ];

        allSVMReg = 2;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0;
    case 'output/ds.set2.dom1',
        allTreeCV = [110 ];
        allMtryCV = [2 ];
        allNodeSizeCV = [20 ];
        
        allSVMReg = .5;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [0.1];
        
        nnCV_AllHid = 2;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0.6;
    case 'output/ds.set2.dom2',
        allTreeCV = [150 ];
        allMtryCV = [60 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = .5;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .2;
        nnCV_regC_Wreg = 0.3;
    case 'output/ds.set3.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [70 ];
        
        allSVMReg = .1;
        
        wFcts = {'welsch'}; %'fair','ols'
        tuningConst = [1.339];
        
        nnCV_AllHid = 1;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0.1;
    case 'output/ds.set4.dom1',
        allTreeCV = [110 ];
        allMtryCV = [70 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = .1;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 2;
        nnCV_regC = .5;
        nnCV_regC_Wreg = 0.1;
    case 'output/ds.set5.dom1',
        allTreeCV = [170 ];
        allMtryCV = [6 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = .5;
        
        wFcts = {'bisquare'}; %'fair','ols'
        tuningConst = [4];
        
        nnCV_AllHid = 1;
        nnCV_regC = .3;
        nnCV_regC_Wreg = 0.4;
    case 'output/ds.set6.dom1',
        allTreeCV = [150 ];
        allMtryCV = [70 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = 2;
        
        wFcts = {'talwar'}; %'fair','ols'
        tuningConst = [2.1];
        
        nnCV_AllHid = 2;
        nnCV_regC = .6;
        nnCV_regC_Wreg = 0.4;
    case 'output/ds.set7.dom1',
        allTreeCV = [150 ];
        allMtryCV = [64 ];
        allNodeSizeCV = [3 ];
        
        allSVMReg = 1;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [0.1];
        
        nnCV_AllHid = 1;
        nnCV_regC = .5;
        nnCV_regC_Wreg = 0.2;
    case 'output/ds.set8.dom1',
        allTreeCV = [170 ];
        allMtryCV = [200 ];
        allNodeSizeCV = [25 ];
        
        allSVMReg = .5;
        
        wFcts = {'huber'}; %'fair','ols'
        tuningConst = [4];
        
        nnCV_AllHid = 1;
        nnCV_regC = .4;
        nnCV_regC_Wreg = 0.1;
    otherwise
        error('different prefix2 than what is implemented?')
end


%% CV Sets for the different parameters
% % FOR CV Only: Boosting
% allTreeCV = [100 110 130 150 170 200];
% allMtryCV = [2 4 6 32 60 64 70 128 160 200 ];
% allNodeSizeCV = [3 20 30 35 50 70 5  25 ];
% allSVMReg = [0.1 .5 1 2] %3 3.5 4 5 10

% % Neural Net:
% nnCV_AllHid = [1 2 3 4];
% nnCV_regC = [0:0.1:0.6]
% nnCV_regC_Wreg = [0:0.1:0.6]

nnCV_AllHid = [1];
nnCV_regC = [0.01]
nnCV_regC_Wreg = [0.3]

%%% that's not necessarily CVing over what we do, which is taking the single neuron as a feature!
% nnCV_AllHid = [1];
% nnCV_regC = [0:0.01:1]
% nnCV_regC_Wreg = [0:0.01:1]


%wFcts = {'andrews','bisquare','cauchy','fair'	,'huber'	,'logistic','ols','talwar','welsch'};
%tuningConst = [0.0001 0.001 .01 0.1,0.3,0.5,0.7,0.9,2.1,2.9,3,4,6, 1.339,4.685,2.385,1.400,	1.345,	1.205,	2.795,2.985];
% best ones only include:
% wFcts = {'bisquare','huber','talwar','welsch'}; %'fair','ols'
% tuningConst = [4,1.339,0.1,1.345,2.1,2.9];


%% Neural Nets
[Y_trn_nn,Y_hat_nn,maxKappaNN,bestParamsNN] = getPrediction_NN(X_trn,Y_trn,X_tst,Y_tst,nnCV_AllHid,nnCV_regC,nnCV_regC_Wreg,prefix);



%% Boosting
[Y_trn_boost,Y_hat_boost,maxKappaBoost] = getPrediction_Boost(X_trn,Y_trn,X_tst,Y_tst,allMtryCV,allTreeCV,allNodeSizeCV,prefix);



%% SVM regression
[Y_trn_svm,Y_hat_svm,maxKappaSVM] = getPrediction_SVM(X_trn,Y_trn,X_tst,Y_tst,allSVMReg,prefix);


%% Linear Regression
% [Y_trn_linReg,Y_hat_linReg,maxKappaLinReg] = getPrediction_linReg(X_trn,Y_trn,X_tst,Y_tst,wFcts,tuningConst,prefix);
[Y_trn_linReg,Y_hat_linReg,maxKappaLinReg] = getPrediction_linReg(X_trn,Y_trn,X_tst,Y_tst,wFcts,tuningConst,prefix,bestParamsNN.model);




%% Bagging!
%%% classifications:
% ntrees = 300;
% Bclass = TreeBagger(ntrees,X_trn,Y_trn)
% Y_hat_bagClass = predict(Bclass,X_tst);
% evalTest = [Y_tst str2double(Y_hat_bagClass)];
% kappaTestBagClass = scoreQuadraticWeightedKappa(evalTest)
%
% ntrees = 160;
% Breg = TreeBagger(ntrees,X_trn,Y_trn,'Method','regression','oobpred','on')
% % plot(oobError(Breg))
% % xlabel('number of grown trees')
% % ylabel('out-of-bag error')
%
% Y_hat_bag = predict(Breg,X_tst);
% evalTest = [Y_tst round(Y_hat_bag)];
% kappaTestReg = scoreQuadraticWeightedKappa(evalTest)
%TODO: Average this too? try




%% Gaussian Process Regression



%% simple average ensemble
maxKappaEns = -Inf;
% % % Mh... this basically just picks the right threshold using our validation-test data, so its improvements are an oracle upper bound
% % % for alpha = 0:0.01:1%
% for alpha = 1/2 % just average all sets %TODO: find on separate set!
%     Y_trn_ens = sum([ alpha*Y_trn_boost (1-alpha)*Y_trn_linReg ],2);
%     Y_hat_ens = sum([ alpha*Y_hat_boost (1-alpha)*Y_hat_linReg ],2);

%%% with neural net: lowers overall kappa, not good
% Y_trn_ens = mean([ Y_trn_boost Y_trn_linReg Y_trn_svm Y_trn_nn],2);
% Y_hat_ens = mean([ Y_hat_boost Y_hat_linReg Y_hat_svm Y_hat_nn],2);

%%% without neural net
Y_trn_ens = mean([ Y_trn_boost Y_trn_linReg Y_trn_svm ],2);
Y_hat_ens = mean([ Y_hat_boost Y_hat_linReg Y_hat_svm ],2);



evalTest = [Y_tst round(Y_hat_ens)];
kappaTest_ens = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest_ens = ' num2str(kappaTest_ens)])
if kappaTest_ens>maxKappaEns
    maxKappaEns =kappaTest_ens
    %bestParamsEns.alpha = alpha;
    %disp(bestParamsEns)
    best_Y_hat = Y_hat_ens;
    best_Y_trn = Y_trn_ens;
end
% end

writeTextFile('_allResults.txt',{prefix 'ens' maxKappaEns },struct('separator', '\t'))
% end




%% write best output to file
allBestKappas = [maxKappaBoost,maxKappaLinReg,maxKappaEns,maxKappaSVM, maxKappaNN]

[winningKappa ind] = max(allBestKappas);
opt.writeFlag = 'w+';

if ind==1
    disp(['Boosting Wins with ' num2str(winningKappa) ])
    writeTextFile(file_trn,Y_trn_boost,opt);
    writeTextFile(file_tst,Y_hat_boost,opt);
elseif ind==2
    disp(['LinReg Wins with ' num2str(winningKappa) ])
    writeTextFile(file_trn,Y_trn_linReg,opt);
    writeTextFile(file_tst,Y_hat_linReg,opt);
elseif ind==3
    disp(['Ensemble Wins with ' num2str(winningKappa) ])
    writeTextFile(file_trn,best_Y_trn,opt);
    writeTextFile(file_tst,best_Y_hat,opt);
elseif ind==4
    disp(['SVM Wins with ' num2str(winningKappa) ])
    writeTextFile(file_trn,Y_trn_svm,opt);
    writeTextFile(file_tst,Y_hat_svm,opt);
elseif ind==5
    disp(['Neural Net Wins with ' num2str(winningKappa) ])
    writeTextFile(file_trn,Y_trn_nn,opt);
    writeTextFile(file_tst,Y_hat_nn,opt);    
else
    error('ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only 4 options!!')
end

disp('DONE: set best winningKappa')




%playAroundTestBed

