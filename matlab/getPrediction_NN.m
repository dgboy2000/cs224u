function [Y_trn_nn,Y_hat_nn,maxKappaNN,bestParamsNN] = getPrediction_NN(X_trn,Y_trn,X_tst,Y_tst,nnCV_AllHid,nnCV_regC,nnCV_regC_Wreg,prefix)

% % good:
% nnCV_AllHid = [1];
% nnCV_regC = [0.01]
% nnCV_regC_Wreg = [0.3]

options.Method = 'lbfgs';
options.MaxIter = 300;
options.Display = 'off'

params.actFunc = 'tanh';
setupActFunc

maxKappaNN = -Inf;
for h = nnCV_AllHid
    for c = nnCV_regC
        for regC = nnCV_regC_Wreg
            
            params.numHid = h;
            params.regC = c;
            params.regC_Wreg = regC;
            
            fanIn = size(X_trn,2);
            range = 1/sqrt(fanIn);
            W = -range + (2*range).*rand(params.numHid,fanIn);
            W(:,end+1) = zeros(params.numHid,1);
            
            % sample regression matrix Wreg
            fanIn = params.numHid;
            range = 1/sqrt(fanIn);
            Wreg = -range + (2*range).*rand(1,params.numHid);
            Wreg(:,end+1) = zeros(1,1);
            
            % options.DerivativeCheck = 'on'
            % X_trn = X_trn(1:30,:);
            % Y_trn = Y_trn(1:30);
            [theta decodeInfo] = param2stack(W,Wreg);
            theta = minFunc(@costFctNN,theta,options,decodeInfo,params,X_trn,Y_trn);
            [W,Wreg] = stack2param(theta, decodeInfo);
            modelNN.W=W;
            modelNN.Wreg=Wreg;
            modelNN.params = params;
            
            Y_trn_hat = predictNN(X_trn,modelNN);
            evalTrain = [Y_trn round(Y_trn_hat)];
            kappaTrain = scoreQuadraticWeightedKappa(evalTrain );
            
            Y_hat_nnCV = predictNN(X_tst,modelNN);
            evalTest = [Y_tst round(Y_hat_nnCV)];
            kappaTestNN = scoreQuadraticWeightedKappa(evalTest);
            disp(['NNet: kappaTrain = ' num2str(kappaTrain) '  kappaTest = ' num2str(kappaTestNN)])
            
            if kappaTestNN >maxKappaNN
                maxKappaNN = kappaTestNN
                bestParamsNN.model = modelNN;
                bestParamsNN.params = params;
                disp(maxKappaNN)
                
                Y_trn_nn = Y_trn_hat;
                Y_hat_nn = Y_hat_nnCV;
            end
            
        end
    end
end
%%
writeTextFile('_allResults.txt',{prefix 'neural' maxKappaNN  bestParamsNN.params.numHid bestParamsNN.params.regC ...
    bestParamsNN.params.regC_Wreg},struct('separator', '\t'))
