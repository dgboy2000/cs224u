function [Y_trn_boost,Y_hat_boost,maxKappaBoost] = getPrediction_Boost(X_trn,Y_trn,X_tst,Y_tst,allMtryCV,allTreeCV,allNodeSizeCV,prefix)
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

Y_trn_boost = regRF_predict(X_trn,bestParams.model);
Y_hat_boost = regRF_predict(X_tst,bestParams.model);
disp(['kappaTest of boosting sanity check= ' num2str(scoreQuadraticWeightedKappa([Y_tst round(Y_hat_boost)]))])




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

