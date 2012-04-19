function [cost,grad] = costFctNN(theta,decodeInfo,params,X,Y)

[W,Wreg] = stack2param(theta, decodeInfo);
N = size(W,1);

X =X';
Y=Y';
%%%%%%%%%%%%%%%%%%%%%%%%%
% forward prop: l = Ab; r = B'a; z = W*f([l ; r]);p = f(z)

p = params.f(W * [X; ones(1,size(X,2))]);
out_a = (Wreg * [p;ones(1,size(p,2))]);
cost = 1/2 * sum(sum((out_a - Y).^2));

delta_out = (out_a-Y);
df_Wreg =  delta_out*[ p' ones(size(p,2),1)];

delta_top = (Wreg' * delta_out) .* params.df([p;ones(1,size(p,2))]);
delta_top = delta_top(1:params.numHid,:);
df_W = delta_top * [X; ones(1,size(X,2))]';

cost = 1/N * cost +  params.regC/2 * sum(W(:).^2) + params.regC_Wreg/2 * sum(Wreg(:).^2);
df_W =  1/N * df_W+ params.regC*W;
df_Wreg=1/N * df_Wreg+ params.regC_Wreg*Wreg;


[grad,~] = param2stack(df_W,df_Wreg);