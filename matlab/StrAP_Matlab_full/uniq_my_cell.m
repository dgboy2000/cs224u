function [B,f1,f2]=uniq_my_cell(idx)
% 
% find unique elements of vector and their frequency
% B ------- the distinct elements
% f1 ------ the occurence times of the elements
% f2 ------ the frequency of the elements

[B,I,J]=unique(idx);

if length(B)>0

for i=1:length(B)
    f1(i)=length(find(J==i));
    f2(i)=f1(i)/length(idx);
end

f1=f1';
f2=f2';

else
    disp('error: input is empty');
end