addpath(genpath('../../../../toolbox'))
addpath(genpath('../../tools'))


%modify so that training data is NxD and labels are Nx1, where N=#of
%examples, D=# of features
allKappas = [];
allMSE  = [];
for d = 2
% for d = 1:8
    X_trn{d} = readTextFile(['data/features/features.set' num2str(d) '.train'],1);
    Y_trn_txt = readTextFile(['data/features/grades.set' num2str(d) '.train']);
    Y_trn{d}  = str2double(Y_trn_txt);
    
    X_tst{d} = readTextFile(['data/features/features.set' num2str(d) '.test'],1);
    Y_tst_txt = readTextFile(['data/features/grades.set' num2str(d) '.test']);
    Y_tst{d}  = str2double(Y_tst_txt);

    %%% this feature normalization only helps set 1,2 a tiny bit
%     m = mean(X_trn{d},1);
%     X_trn{d} = bsxfun(@minus,X_trn{d},m);
%     stdev = 1./std(X_trn{d},1);
%     stdev(isinf(stdev)) = 1;
%     X_trn{d} = bsxfun(@times,stdev,X_trn{d});
%     X_trn{d} = bsxfun(@max,bsxfun(@min,X_trn{d},3*stdev),-3*stdev);
%     X_tst{d} = bsxfun(@minus,X_tst{d},m);
%     X_tst{d} = bsxfun(@times,stdev,X_tst{d});
%     X_tst{d} = bsxfun(@max,bsxfun(@min,X_tst{d},3*stdev),-3*stdev);
    
    
    % example 1:  simply use with the defaults
    model = regRF_train(X_trn{d},Y_trn{d});
    Y_hat_trn = regRF_predict(X_trn{d},model);
    opt.writeFlag = 'w+';
    writeTextFile(['../../../data/features/matOut.grades.set' num2str(d) '.train'],Y_hat_trn,opt);
    evalTrain = [Y_tst{d} round(Y_hat)];
    kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
    
    
    Y_hat = regRF_predict(X_tst{d},model);
    writeTextFile(['../../../data/features/matOut.grades.set' num2str(d) '.test'],Y_hatopt,opt)
    
    mse = sum((Y_hat-Y_tst{d}).^2);
    fprintf('\nexample 1: MSE rate %f\n', mse  );
    allMSE = [allMSE mse];
    evalTest = [Y_tst{d} round(Y_hat)];
    kappa = scoreQuadraticWeightedKappa(evalTest);
    allKappas = [allKappas kappa]
end

meanQuadraticWeightedKappa(allKappas)




% 
% % try 3 and 4 together to predict only 3 or only 4 -> hurts performance :/
% %%
% model = regRF_train([X_trn{3};X_trn{4}],[Y_trn{3};Y_trn{4}]);
% d=4
% Y_hat = regRF_predict(X_tst{d},model);
% mse = sum((Y_hat-Y_tst{d}).^2);
% fprintf('\nexample 1: MSE rate %f\n', mse  );
% evalTest = [Y_tst{d} round(Y_hat)];
% kappa = scoreQuadraticWeightedKappa(evalTest)
% 
% % try 5 and 6 together to predict only 5 or only 6 -> helped 5 a little bit (0.8077 kappa vs 0.7961 or best before 0.8005)
% %%
% model = regRF_train([X_trn{5};X_trn{6}],[Y_trn{5};Y_trn{6}]);
% d=5
% Y_hat = regRF_predict(X_tst{d},model);
% mse = sum((Y_hat-Y_tst{d}).^2);
% fprintf('\nexample 1: MSE rate %f\n', mse  );
% evalTest = [Y_tst{d} round(Y_hat)];
% kappa = scoreQuadraticWeightedKappa(evalTest)
% 
% % try 7 and 8 together to predict only 8 -> hacky with 2* values from 7, hurts performance
% %%
% model = regRF_train([X_trn{7};X_trn{8}],[2*Y_trn{7};Y_trn{8}]);
% d=8
% Y_hat = regRF_predict(X_tst{d},model);
% mse = sum((Y_hat-Y_tst{d}).^2);
% fprintf('\nexample 1: MSE rate %f\n', mse  );
% evalTest = [Y_tst{d} round(Y_hat)];
% kappa = scoreQuadraticWeightedKappa(evalTest)