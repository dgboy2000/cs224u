function [d,f1,f2]=uniq_my(idx)
% 
% find unique elements of vector and their frequency
% d ------- the distinct elements
% f1 ------ the occurence times of the elements
% f2 ------ the frequency of the elements

if ~isempty(idx)
   d=unique(idx);
   for i=1:length(d)
       f1(i)=length(find(idx==d(i)));
       f2(i)=f1(i)/length(idx);
   end

   f1=f1';
   f2=f2';

else
    disp('error: input is empty');
end