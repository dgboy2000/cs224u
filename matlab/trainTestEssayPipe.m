
%prefix='output/features.set1.dom1';prefix2='output/ds.set1.dom1';
%matlab/trainTestEssayPipe


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

[X_trn Y_trn] = getFeatAndGrade(prefix,prefix2,'train');
[X_tst Y_tst] = getFeatAndGrade(prefix,prefix2,'test');

model = regRF_train(X_trn,Y_trn);
Y_trn_hat = regRF_predict(X_trn,model);
opt.writeFlag = 'w+';
writeTextFile([prefix2 '.train' '.matOut'],Y_trn_hat,opt);
evalTrain = [Y_trn round(Y_trn_hat)];
kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
disp(['kappaTrain = ' num2str(kappaTrain)])

Y_hat = regRF_predict(X_tst,model);
writeTextFile([prefix2 '.test' '.matOut'],Y_hat,opt)

mse = sum((Y_hat-Y_tst).^2);
fprintf('\nexample 1: MSE rate %f\n', mse  );
evalTest = [Y_tst round(Y_hat)];
kappaTest = scoreQuadraticWeightedKappa(evalTest);
disp(['kappaTest = ' num2str(kappaTest)])
