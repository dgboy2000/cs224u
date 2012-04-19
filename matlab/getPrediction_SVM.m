function [Y_trn_svm,Y_hat_svm,maxKappaSVM] = getPrediction_SVM(X_trn,Y_trn,X_tst,Y_tst,allSVMReg,prefix);

maxKappaSVM = -Inf;
for c = allSVMReg
    modelSVM = svmtrain(Y_trn, X_trn, ['-s 3 -t 0 -c ' num2str(c)]);
    [Y_trn_svm, ~, ~] = svmpredict(Y_trn, X_trn, modelSVM);
    [Y_hat_svm, ~, ~] = svmpredict(Y_tst, X_tst, modelSVM);
    evalTest = [Y_tst round(Y_hat_svm)];
    kappaTestSVM = scoreQuadraticWeightedKappa(evalTest);
    
    if kappaTestSVM >maxKappaSVM
        maxKappaSVM = kappaTestSVM
        bestParamsSVM.model = modelSVM;
        bestParamsSVM.c = c;
        
        [Y_trn_svm, ~, ~] = svmpredict(Y_trn, X_trn, bestParamsSVM.model);
        [Y_hat_svm, ~, ~] = svmpredict(Y_tst, X_tst, bestParamsSVM.model); 

        disp(bestParamsSVM)
    end
    
end
writeTextFile('_allResults.txt',{prefix 'svm' maxKappaSVM  bestParamsSVM.c},struct('separator', '\t'))