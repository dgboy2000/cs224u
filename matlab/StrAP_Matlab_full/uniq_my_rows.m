function [uu,idx,f,alll]=uniq_my_rows(x)
% 
% find unique rows and their frequency
% uu ------- the distinct rows
% idx ------ the position indicator of distinct rows in x ( uu=x(idx,:) )
% f ------ the occurence times of the elements
% alll ------ uu(alll,:) = x

if ~isempty(x)
   [uu,idx,alll]=unique(x,'rows');
   for i=1:max(alll)
       f(i)=length(find(alll==i));
   end

   f=f';
else
   disp('error: input is empty');
end
