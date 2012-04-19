function Y_hat_nn = predictNN(X_tst,model)

p = model.params.f(model.W * [X_tst'; ones(1,size(X_tst',2))]);
Y_hat_nn = (model.Wreg * [p;ones(1,size(p,2))]);
Y_hat_nn =Y_hat_nn';
