
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


% [X_trn_text,X_tst_text] = extractTextFeatures(Text_trn,Text_tst,prefix2);
% X_trn = [X_trn X_trn_text];
% X_tst = [X_tst X_tst_text];

%save('allDataSet2Dom2.mat','X_trn','Y_trn','Text_trn','X_tst','Y_tst','Text_tst','X_trn_text','X_tst_text')

m = mean(X_trn,1);
X_trn = bsxfun(@minus,X_trn,m);
stdev = 1./std(X_trn,1);
stdev(isinf(stdev)) = 1;
X_trn = bsxfun(@times,stdev,X_trn);
%X_trn = bsxfun(@max,bsxfun(@min,X_trn,3*stdev),-3*stdev);
X_tst = bsxfun(@minus,X_tst,m);
X_tst = bsxfun(@times,stdev,X_tst);
%X_tst = bsxfun(@max,bsxfun(@min,X_tst,3*stdev),-3*stdev);



%% original simple single boosting
% model = regRF_train(X_trn,Y_trn);
% Y_trn_hat = regRF_predict(X_trn,model);
% opt.writeFlag = 'w+';
% %writeTextFile([prefix2 '.train' '.matout'],Y_trn_hat,opt);
% evalTrain = [Y_trn round(Y_trn_hat)];
% kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
% disp(['kappaTrain = ' num2str(kappaTrain)])
%
% Y_hat = regRF_predict(X_tst,model);
% %writeTextFile([prefix2 '.test' '.matout'],Y_hat,opt)
%
% mse = sum((Y_hat-Y_tst).^2);
% fprintf('\nexample 1: MSE rate %f\n', mse  );
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])


%% CV over boosting (sometimes a good linear regression performs much better!)
% different CV sets for diff. essay sets
switch prefix2
    case 'output/ds.set1.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [5 ];
    case 'output/ds.set2.dom1',
        allTreeCV = [110 ];
        allMtryCV = [2 ];
        allNodeSizeCV = [20 ];
    case 'output/ds.set2.dom2',
        % SUCKS:   maxKappaBoost=0.65798 :/
        allTreeCV = [150 ];
        allMtryCV = [60 ];
        allNodeSizeCV = [25 ];
    case 'output/ds.set3.dom1',
        allTreeCV = [130 ];
        allMtryCV = [160 ];
        allNodeSizeCV = [70 ];
    case 'output/ds.set4.dom1',
        allTreeCV = [110 ];
        allMtryCV = [70 ];
        allNodeSizeCV = [3 ];
    case 'output/ds.set5.dom1',
        allTreeCV = [170 ];
        allMtryCV = [6 ];
        allNodeSizeCV = [3 ];
    case 'output/ds.set6.dom1',
        allTreeCV = [150 ];
        allMtryCV = [70 ];
        allNodeSizeCV = [25 ];
    case 'output/ds.set7.dom1',
        allTreeCV = [150 ];
        allMtryCV = [64 ];
        allNodeSizeCV = [3 ];
    case 'output/ds.set8.dom1',
        allTreeCV = [170 ];
        allMtryCV = [200 ];
        allNodeSizeCV = [25 ];
    otherwise
        error('different prefix2 than what is implemented?')
end

% % FOR CV Only:
% allTreeCV = [100 110 130 150 170 200];
%     allMtryCV = [2 4 6 32 60 64 70 128 160 200 ];
%     allNodeSizeCV = [3 20 30 35 50 70 5  25 ];


maxKappaBoost = -Inf;
for imp = 1 % 0:1
    %for bias=0:1
    for bias=1
        %for numTrees = [50 70 90 100 110 150 200]
        for mtry = allMtryCV
            for numTrees = allTreeCV
                %for mtry = [2 3 4 5 8 16 32 64 128]
                for nodeSize = allNodeSizeCV
                    
                    clear extra_options
                    extra_options.corr_bias=bias;
                    extra_options.importance = imp; %(0 = (Default) Don't, 1=calculate)
                    extra_options.nodesize =nodeSize;
                    %mtry = 4; %default 4: % mtry  = number of predictors sampled for spliting at each node.
                    model = regRF_train(X_trn,Y_trn, numTrees, mtry, extra_options);
                    
                    Y_trn_hat = regRF_predict(X_trn,model);
                    evalTrain = [Y_trn round(Y_trn_hat)];
                    kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
                    
                    Y_hat = regRF_predict(X_tst,model);
                    %fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
                    evalTest = [Y_tst round(Y_hat)];
                    kappaTest = scoreQuadraticWeightedKappa(evalTest);
                    disp(['kappaTrain = ' num2str(kappaTrain) '  kappaTest = ' num2str(kappaTest)])
                    if kappaTest > maxKappaBoost
                        %New best!
                        maxKappaBoost=kappaTest;
                        bestParams.numTrees = numTrees;
                        bestParams.mtry = mtry;
                        
                        bestParams.nodeSize=nodeSize;
                        bestParams.importance=imp;
                        bestParams.corr_bias = bias;
                        
                        bestParams.model = model;
                        
                        disp(bestParams)
                    end
                end
            end
        end
    end
end

%% Best Boosting Result
disp(prefix)
%disp(prefix2)
% maxKappaBoost = maxKappa;
disp(['maxKappaBoost=' num2str(maxKappaBoost)])
disp(bestParams)
writeTextFile('_allResults.txt',{prefix 'boost' maxKappaBoost bestParams.mtry  bestParams.numTrees ...
    bestParams.nodeSize},struct('separator', '\t')) %bestParams.corr_bias bestParams.importance always 1


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




%% libSVM: add to ensemble
maxKappaSVM = -Inf;
for c = [0.1 .5 1 2 3 3.5 4 5 ]
    modelSVM = svmtrain(Y_trn, X_trn, ['-s 3 -t 0 -c ' num2str(c)]);
    [Y_trn_svm, accuracy, dec_values] = svmpredict(Y_trn, X_trn, modelSVM); % test the training data
    [Y_hat_svm, accuracy, dec_values] = svmpredict(Y_tst, X_tst, modelSVM); % test the training data
    evalTest = [Y_tst round(Y_hat_svm)];
    kappaTestSVM = scoreQuadraticWeightedKappa(evalTest)
    
    if kappaTestSVM >maxKappaSVM
        maxKappaSVM = kappaTestSVM
        bestParamsSVM.model = modelSVM;
        bestParamsSVM.c = c;
        disp(bestParamsSVM)
    end
    
end
writeTextFile('_allResults.txt',{prefix 'svm' maxKappaSVM  bestParamsSVM.c},struct('separator', '\t'))


%% CV over linear regression

wFcts = {'andrews','bisquare','cauchy','fair'	,'huber'	,'logistic','ols','talwar','welsch'};
tuningConst = [0.0001 0.001 .01 0.1,0.3,0.5,0.7,0.9,2.1,2.9,3,4,6, 1.339,4.685,2.385,1.400,	1.345,	1.205,	2.795,2.985];

maxKappaLinReg = -Inf;
for w = wFcts
    for t = tuningConst
        b = robustfit(X_trn,Y_trn,w{1},t);
        Y_trn_hat = [ones(size(X_trn,1),1) X_trn]*b;
        evalTrain = [Y_trn round(Y_trn_hat)];
        kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
        %disp(['kappaTrain = ' num2str(kappaTrain)])
        
        Y_hat= [ones(size(X_tst,1),1) X_tst]*b;
        %fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
        evalTest = [Y_tst round(Y_hat)];
        kappaTest = scoreQuadraticWeightedKappa(evalTest);
        disp(['kappaTrain = ' num2str(kappaTrain) ' kappaTest = ' num2str(kappaTest)])
        if kappaTest>maxKappaLinReg
            maxKappaLinReg = kappaTest;
            bestParamsLinReg.w = w;
            bestParamsLinReg.t = t;
            bestParamsLinReg.b = b;
            disp(bestParamsLinReg)
        end
    end
end
writeTextFile('_allResults.txt',{prefix 'linReg' maxKappaLinReg bestParamsLinReg.w{1} bestParamsLinReg.t},struct('separator', '\t'))

%ensemble only if they're sort of close-by, ow, one seems to hurt the other
% if abs(maxKappaLinReg-maxKappaBoost)<0.01



%% simple average ensemble
Y_trn_boost = regRF_predict(X_trn,bestParams.model);
Y_hat_boost = regRF_predict(X_tst,bestParams.model);
disp(['kappaTest of boosting sanity check= ' scoreQuadraticWeightedKappa([Y_tst round(Y_hat_boost)])])

% [b,STATS] = robustfit(X_trn,Y_trn,bestParamsLinReg.w{1},bestParamsLinReg.t);
Y_trn_linReg = [ones(size(X_trn,1),1) X_trn]*bestParamsLinReg.b;
Y_hat_linReg = [ones(size(X_tst,1),1) X_tst]*bestParamsLinReg.b;

[Y_trn_svm, accuracy, dec_values] = svmpredict(Y_trn, X_trn, bestParamsSVM.model); % test the training data
[Y_hat_svm, accuracy, dec_values] = svmpredict(Y_tst, X_tst, bestParamsSVM.model); % test the training data


maxKappaEns = -Inf;
% % % Mh... this basically just picks the right threshold using our validation-test data, so its improvements are an oracle upper bound
% % % for alpha = 0:0.01:1%
% for alpha = 1/2 % just average all sets %TODO: find on separate set!
%     Y_trn_ens = sum([ alpha*Y_trn_boost (1-alpha)*Y_trn_linReg ],2);
%     Y_hat_ens = sum([ alpha*Y_hat_boost (1-alpha)*Y_hat_linReg ],2);

Y_trn_ens = mean([ Y_trn_boost Y_trn_linReg Y_trn_svm],2);
Y_hat_ens = mean([ Y_hat_boost Y_hat_linReg Y_hat_svm],2);

evalTest = [Y_tst round(Y_hat_ens)];
kappaTest_ens = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest_ens = ' num2str(kappaTest)])
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
allBestKappas = [maxKappaBoost,maxKappaLinReg,maxKappaEns,maxKappaSVM]
[val ind] = max(allBestKappas);
opt.writeFlag = 'a+';

if ind==1
    disp(['Boosting Wins with ' num2str(val) ])
    writeTextFile(file_trn,Y_trn_boost,opt);
    writeTextFile(file_tst,Y_hat_boost,opt);
elseif ind==2
    disp(['LinReg Wins with ' num2str(val) ])
    writeTextFile(file_trn,Y_trn_linReg,opt);
    writeTextFile(file_tst,Y_hat_linReg,opt);
elseif ind==3
    disp(['Ensemble Wins with ' num2str(val) ])
    writeTextFile(file_trn,best_Y_trn,opt);
    writeTextFile(file_tst,best_Y_hat,opt);
elseif ind==4
    disp(['SVM Wins with ' num2str(val) ])
    writeTextFile(file_trn,Y_trn_svm,opt);
    writeTextFile(file_tst,Y_hat_svm,opt);
else
    error('ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only 4 options!!')
end


disp('DONE.')




%%
%%%%%%%%%%%%%%%% play around:
%
% close all
% % example 2:  set to 100 trees
% model = regRF_train(X_trn,Y_trn, 100);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 2: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
%
%
% % example 3:  set to 100 trees, mtry = 2
% model = regRF_train(X_trn,Y_trn, 100,2);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 3: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
%
% % example 4:  set to defaults trees and mtry by specifying values as 0
% model = regRF_train(X_trn,Y_trn, 0, 0);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 4: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
%
% % % example 5: set sampling without replacement (default is with replacement)
% extra_options.replace = 0 ;
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 5: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% % example 6: sampsize example
% %  extra_options.sampsize =  Size(s) of sample to draw. For classification,
% %                   if sampsize is a vector of the length the number of strata, then sampling is stratified by strata,
% %                   and the elements of sampsize indicate the numbers to be drawn from the strata.
% clear extra_options
% extra_options.sampsize = size(X_trn,1)*2/3;
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 6: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% % example 7: nodesize
% %  extra_options.nodesize = Minimum size of terminal nodes. Setting this number larger causes smaller trees
% %                   to be grown (and thus take less time). Note that the default values are different
% %                   for classification (1) and regression (5).
% clear extra_options
% extra_options.nodesize = 7;
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 7: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% % example 8: calculating importance
% clear extra_options
% extra_options.importance = 1; %(0 = (Default) Don't, 1=calculate)
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% %model will have 3 variables for importance importanceSD and localImp
% %importance = a matrix with nclass + 2 (for classification) or two (for regression) columns.
% %           For classification, the first nclass columns are the class-specific measures
% %           computed as mean decrease in accuracy. The nclass + 1st column is the
% %           mean decrease in accuracy over all classes. The last column is the mean decrease
% %           in Gini index. For Regression, the first column is the mean decrease in
% %           accuracy and the second the mean decrease in MSE. If importance=FALSE,
% %           the last measure is still returned as a vector.
% figure('Name','Importance Plots')
% subplot(3,1,1);
% bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
% title('Mean decrease in Accuracy');
%
% subplot(3,1,2);
% bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
% title('Mean decrease in Gini index');
%
%
% %importanceSD = The ?standard errors? of the permutation-based importance measure. For classification,
% %           a D by nclass + 1 matrix corresponding to the first nclass + 1
% %           columns of the importance matrix. For regression, a length p vector.
% model.importanceSD;
% subplot(3,1,3);
% bar(model.importanceSD);xlabel('feature');ylabel('magnitude');
% title('Std. errors of importance measure');
%
% % example 9: calculating local importance
% %  extra_options.localImp = Should casewise importance measure be computed? (Setting this to TRUE will
% %                   override importance.)
% %localImp  = a D by N matrix containing the casewise importance measures, the [i,j] element
% %           of which is the importance of i-th variable on the j-th case. NULL if
% %          localImp=FALSE.
% clear extra_options
% extra_options.localImp = 1; %(0 = (Default) Don't, 1=calculate)
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 9: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% % example 10: calculating proximity
% %  extra_options.proximity = Should proximity measure among the rows be calculated?
% clear extra_options
% extra_options.proximity = 1; %(0 = (Default) Don't, 1=calculate)
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 10: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
%
% % example 11: use only OOB for proximity
% %  extra_options.oob_prox = Should proximity be calculated only on 'out-of-bag' data?
% clear extra_options
% extra_options.proximity = 1; %(0 = (Default) Don't, 1=calculate)
% extra_options.oob_prox = 0; %(Default = 1 if proximity is enabled,  Don't 0)
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 11: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% % example 12: to see what is going on behind the scenes
% %  extra_options.do_trace = If set to TRUE, give a more verbose output as randomForest is run. If set to
% %                   some integer, then running output is printed for every
% %                   do_trace trees.
% clear extra_options
% extra_options.do_trace = 1; %(Default = 0)
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 12: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
%
% % example 13: to see what is going on behind the scenes
% %  extra_options.keep_inbag Should an n by ntree matrix be returned that keeps track of which samples are
% %                   'in-bag' in which trees (but not how many times, if sampling with replacement)
%
% clear extra_options
% extra_options.keep_inbag = 1; %(Default = 0)
%
% model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 13: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
%
% %
% % example 15: nPerm
% %               Number of times the OOB data are permuted per tree for assessing variable
% %               importance. Number larger than 1 gives slightly more stable estimate, but not
% %               very effective. Currently only implemented for regression.
% clear extra_options
% extra_options.importance=1;
% extra_options.nPerm = 1; %(Default = 0)
% model = regRF_train(X_trn,Y_trn,100,2,extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 15: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
%
% figure('Name','Importance Plots nPerm=1')
% subplot(2,1,1);
% bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
% title('Mean decrease in Accuracy');
%
% subplot(2,1,2);
% bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
% title('Mean decrease in Gini index');
%
% %let's now run with nPerm=3
% clear extra_options
% extra_options.importance=1;
% extra_options.nPerm = 3; %(Default = 0)
% model = regRF_train(X_trn,Y_trn,100,2,extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 15: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
% figure('Name','Importance Plots nPerm=3')
% subplot(2,1,1);
% bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
% title('Mean decrease in Accuracy');
%
% subplot(2,1,2);
% bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
% title('Mean decrease in Gini index');
%
% % example 16: corr_bias (not recommended to use)
% clear extra_options
% extra_options.corr_bias=1;
% model = regRF_train(X_trn,Y_trn,100,2,extra_options);
% Y_hat = regRF_predict(X_tst,model);
% fprintf('\nexample 16: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% play with ridge regression: FAIL :/
% % b = ridge(Y_trn,X_trn,1);
% b = inv((X_trn'*X_trn+eye(size(X_trn,2))))*X_trn'*Y_trn;
% % Y_trn_hat = [ones(size(X_trn,1),1) X_trn]*b;
% Y_trn_hat = [X_trn]*b;
% evalTrain = [Y_trn round(Y_trn_hat)];
% kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
% %disp(['kappaTrain = ' num2str(kappaTrain)])
%
%
% % Y_hat= [ones(size(X_tst,1),1) X_tst]*b;
% Y_hat= [X_tst]*b;
% %fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
% evalTest = [Y_tst round(Y_hat)];
% kappaTest = scoreQuadraticWeightedKappa(evalTest);
% disp(['kappaTest = ' num2str(kappaTest)])









