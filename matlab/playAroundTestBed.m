%
%%%%%%%%%%%%%%% play around:

close all
% example 2:  set to 100 trees
model = regRF_train(X_trn,Y_trn, 100);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 2: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])



% example 3:  set to 100 trees, mtry = 2
model = regRF_train(X_trn,Y_trn, 100,2);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 3: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])


% example 4:  set to defaults trees and mtry by specifying values as 0
model = regRF_train(X_trn,Y_trn, 0, 0);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 4: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])


% % example 5: set sampling without replacement (default is with replacement)
extra_options.replace = 0 ;
model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 5: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

% example 6: sampsize example
%  extra_options.sampsize =  Size(s) of sample to draw. For classification,
%                   if sampsize is a vector of the length the number of strata, then sampling is stratified by strata,
%                   and the elements of sampsize indicate the numbers to be drawn from the strata.
clear extra_options
extra_options.sampsize = size(X_trn,1)*2/3;

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 6: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

% example 7: nodesize
%  extra_options.nodesize = Minimum size of terminal nodes. Setting this number larger causes smaller trees
%                   to be grown (and thus take less time). Note that the default values are different
%                   for classification (1) and regression (5).
clear extra_options
extra_options.nodesize = 7;

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 7: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

% example 8: calculating importance
clear extra_options
extra_options.importance = 1; %(0 = (Default) Don't, 1=calculate)

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

%model will have 3 variables for importance importanceSD and localImp
%importance = a matrix with nclass + 2 (for classification) or two (for regression) columns.
%           For classification, the first nclass columns are the class-specific measures
%           computed as mean decrease in accuracy. The nclass + 1st column is the
%           mean decrease in accuracy over all classes. The last column is the mean decrease
%           in Gini index. For Regression, the first column is the mean decrease in
%           accuracy and the second the mean decrease in MSE. If importance=FALSE,
%           the last measure is still returned as a vector.
figure('Name','Importance Plots')
subplot(3,1,1);
bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
title('Mean decrease in Accuracy');

subplot(3,1,2);
bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
title('Mean decrease in Gini index');


%importanceSD = The ?standard errors? of the permutation-based importance measure. For classification,
%           a D by nclass + 1 matrix corresponding to the first nclass + 1
%           columns of the importance matrix. For regression, a length p vector.
model.importanceSD;
subplot(3,1,3);
bar(model.importanceSD);xlabel('feature');ylabel('magnitude');
title('Std. errors of importance measure');

% example 9: calculating local importance
%  extra_options.localImp = Should casewise importance measure be computed? (Setting this to TRUE will
%                   override importance.)
%localImp  = a D by N matrix containing the casewise importance measures, the [i,j] element
%           of which is the importance of i-th variable on the j-th case. NULL if
%          localImp=FALSE.
clear extra_options
extra_options.localImp = 1; %(0 = (Default) Don't, 1=calculate)

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 9: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

% example 10: calculating proximity
%  extra_options.proximity = Should proximity measure among the rows be calculated?
clear extra_options
extra_options.proximity = 1; %(0 = (Default) Don't, 1=calculate)

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 10: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])


% example 11: use only OOB for proximity
%  extra_options.oob_prox = Should proximity be calculated only on 'out-of-bag' data?
clear extra_options
extra_options.proximity = 1; %(0 = (Default) Don't, 1=calculate)
extra_options.oob_prox = 0; %(Default = 1 if proximity is enabled,  Don't 0)

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 11: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

% example 12: to see what is going on behind the scenes
%  extra_options.do_trace = If set to TRUE, give a more verbose output as randomForest is run. If set to
%                   some integer, then running output is printed for every
%                   do_trace trees.
clear extra_options
extra_options.do_trace = 1; %(Default = 0)

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 12: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));

% example 13: to see what is going on behind the scenes
%  extra_options.keep_inbag Should an n by ntree matrix be returned that keeps track of which samples are
%                   'in-bag' in which trees (but not how many times, if sampling with replacement)

clear extra_options
extra_options.keep_inbag = 1; %(Default = 0)

model = regRF_train(X_trn,Y_trn, 100, 4, extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 13: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])


%
% example 15: nPerm
%               Number of times the OOB data are permuted per tree for assessing variable
%               importance. Number larger than 1 gives slightly more stable estimate, but not
%               very effective. Currently only implemented for regression.
clear extra_options
extra_options.importance=1;
extra_options.nPerm = 1; %(Default = 0)
model = regRF_train(X_trn,Y_trn,100,2,extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 15: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));

figure('Name','Importance Plots nPerm=1')
subplot(2,1,1);
bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
title('Mean decrease in Accuracy');

subplot(2,1,2);
bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
title('Mean decrease in Gini index');

%let's now run with nPerm=3
clear extra_options
extra_options.importance=1;
extra_options.nPerm = 3; %(Default = 0)
model = regRF_train(X_trn,Y_trn,100,2,extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 15: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

figure('Name','Importance Plots nPerm=3')
subplot(2,1,1);
bar(model.importance(:,end-1));xlabel('feature');ylabel('magnitude');
title('Mean decrease in Accuracy');

subplot(2,1,2);
bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
title('Mean decrease in Gini index');

% example 16: corr_bias (not recommended to use)
clear extra_options
extra_options.corr_bias=1;
model = regRF_train(X_trn,Y_trn,100,2,extra_options);
Y_hat = regRF_predict(X_tst,model);
fprintf('\nexample 16: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% quick hack for lin reg combining all outputs
ensXtrn = [ Y_trn_boost Y_trn_linReg Y_trn_svm];
ensXtst = [ Y_hat_boost Y_hat_linReg Y_hat_svm];
wFcts = {'ols'};
tuningConst = [1];

maxKappaLinReg = -Inf;
for w = wFcts
    for t = tuningConst

        b = robustfit(ensXtrn,Y_trn,w{1},t)
        Y_trn_hat = [ones(size(ensXtrn,1),1) ensXtrn]*b;
        evalTrain = [Y_trn round(Y_trn_hat)];
        kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
        %disp(['kappaTrain = ' num2str(kappaTrain)])

        Y_hat= [ones(size(ensXtst,1),1) ensXtst]*b;
        %fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
        evalTest = [Y_tst round(Y_hat)];
        kappaTest = scoreQuadraticWeightedKappa(evalTest);
        disp(['kappaTrain = ' num2str(kappaTrain) ' kappaTest = ' num2str(kappaTest)])
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ridge regression
tuningConst = [0:0.1:20];
%tuningConst = [17];

maxKappaRidgeReg = -Inf;
for t = tuningConst
    b = ridge(Y_trn,X_trn,t,0);
    Y_trn_hat = [ones(size(X_trn,1),1) X_trn]*b;
    evalTrain = [Y_trn round(Y_trn_hat)];
    kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
    %disp(['kappaTrain = ' num2str(kappaTrain)])

    Y_hat= [ones(size(X_tst,1),1) X_tst]*b;
    %fprintf('\nexample 8: MSE rate %f\n',   sum((Y_hat-Y_tst).^2));
    evalTest = [Y_tst round(Y_hat)];
    kappaTest = scoreQuadraticWeightedKappa(evalTest);
    disp(['kappaTrain = ' num2str(kappaTrain) ' kappaTest = ' num2str(kappaTest)])
    if kappaTest>maxKappaRidgeReg
        maxKappaRidgeReg = kappaTest;
        bestParamsRidgeReg.t = t;
        disp(bestParamsRidgeReg)
    end
end

writeTextFile('_allResults.txt',{prefix 'ridgeReg' maxKappaRidgeReg  bestParamsRidgeReg.t},struct('separator', '\t'))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%