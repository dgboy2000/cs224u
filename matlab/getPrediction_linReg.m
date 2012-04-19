function [Y_trn_linReg,Y_hat_linReg,maxKappaLinReg] = getPrediction_linReg(X_trn,Y_trn,X_tst,Y_tst,wFcts,tuningConst,prefix,modelNN)

if exist('modelNN','var')
    useNeuronsAsFeatures=1
else
    useNeuronsAsFeatures=0
end

if useNeuronsAsFeatures == 1
    
    maxKappaLinReg = -Inf;
    for w = wFcts
        for t = tuningConst
            p_trn = modelNN.params.f(modelNN.W * [X_trn'; ones(1,size(X_trn',2))]);
            p_tst = modelNN.params.f(modelNN.W * [X_tst'; ones(1,size(X_tst',2))]);
            
            b = robustfit([X_trn p_trn'],Y_trn,w{1},t);
            Y_trn_hat = [ones(size(X_trn,1),1) [X_trn p_trn']]*b;
            evalTrain = [Y_trn round(Y_trn_hat)];
            kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
            %disp(['kappaTrain = ' num2str(kappaTrain)])
            
            Y_hat= [ones(size(X_tst,1),1) [X_tst p_tst']]*b;
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
                
                Y_trn_linReg = Y_trn_hat;
                Y_hat_linReg = Y_hat;
            end
        end
    end
    disp(['Best Kappa linReg: ' num2str(maxKappaLinReg)])
    
    writeTextFile('_allResults.txt',{prefix 'linReg' maxKappaLinReg bestParamsLinReg.w{1} bestParamsLinReg.t},struct('separator', '\t'))
    
else
    maxKappaLinReg = -Inf;
    for w = wFcts
        for t = tuningConst
            
            b = robustfit([X_trn ],Y_trn,w{1},t);
            Y_trn_hat = [ones(size(X_trn,1),1) [X_trn ]]*b;
            evalTrain = [Y_trn round(Y_trn_hat)];
            kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
            %disp(['kappaTrain = ' num2str(kappaTrain)])
            
            Y_hat= [ones(size(X_tst,1),1) [X_tst ]]*b;
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
                
                Y_trn_linReg = Y_trn_hat;
                Y_hat_linReg = Y_hat;
            end
        end
    end
    disp(['Best Kappa linReg: ' num2str(maxKappaLinReg)])
    
    writeTextFile('_allResults.txt',{prefix 'linReg' maxKappaLinReg bestParamsLinReg.w{1} bestParamsLinReg.t},struct('separator', '\t'))
    
end