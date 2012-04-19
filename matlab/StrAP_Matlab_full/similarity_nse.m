function [ss]=similarity_nse(x)
%
% obtain the negative squared error(Euclidean distance) ss
% for each pair of examples
 

for i=1:size(x,1)
    for j=(i+1):size(x,1)
        ss(i,j)=-norm(x(i,:)-x(j,:))^2;
        ss(j,i)=ss(i,j);
    end
end
