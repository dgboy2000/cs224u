function R=com_accu(label,label_given)
%%% compare true label with given label
N=length(label_given);
R=zeros(N,1);

if isnumeric(label)
    for i=1:N
        if (label(i)==label_given(i))
            R(i)=1;
        elseif (label_given(i)==-1)
            R(i)=-1;
        end
    end
else
    for i=1:N
    	tline=label(i);
    	if (strcmp(tline, label_given(i))==1)
          R(i)=1;
    	elseif  (cell2mat(label_given(i))==-1)
          R(i)=-1;
    	end
    end
end
