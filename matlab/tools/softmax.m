function x = softmax(x)
%Softmax matrix x (columns sum to 1).
%2010, richard @ socher.org
x = exp(x);
x = bsxfun(@ldivide,sum(x),x);
