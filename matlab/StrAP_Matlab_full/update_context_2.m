function temp=update_context_2(tempold,add)
    temp=tempold;
    if isnumeric(add)
        for j=1:size(add,1)
            not_find=1;
            for t=1:size(tempold,1)
                if (add(j,1)==tempold(t,1))
                    not_find=0;
                    temp(t,2)=tempold(t,2)+add(j,2);
                    break
                end
            end

            if not_find
                temp=[temp;add(j,:)];
            end
 
        end

    else

    for j=1:size(add,1) 
        not_find=1;
        for t=1:size(tempold,1)
            if (strcmp(add(j,1), tempold(t,1))==1)
                not_find=0;
                temp(t,2)={cell2mat(tempold(t,2)) + cell2mat(add(j,2))};
                break
            end
        end

        if not_find
           temp=[temp; add(j,:)];
        end
    end

    end