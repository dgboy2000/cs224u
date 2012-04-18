function temp=update_context(cluster,XX,XX_V,ii)
    temp=getfield(cluster,XX);
    
    if isnumeric(XX_V)
        if (~isempty(temp) && (length(temp) >= ii) )
            temp=temp{ii};
            not_find=1;
                for j=1:size(temp,1)
                    if (temp(j,1) == XX_V)
                        not_find=0;
                        temp(j,2)=temp(j,2)+1;
                        break
                    end
                end

            if not_find
                temp=[temp; [XX_V 1]];
            end

        else
            temp=[XX_V 1];
        end

    else
        if (~isempty(temp) && (length(temp) >= ii) )
            temp=temp{ii};
            not_find=1;
            for j=1:size(temp,1)
                if (strcmp(temp(j,1),XX_V)==1)
                    not_find=0;
                    temp(j,2)={cell2mat(temp(j,2))+1};
                    break
                end
            end

            if not_find
                temp=[temp; [XX_V num2cell(1)]];
            end

        else
            temp=[XX_V num2cell(1)];
        end

    end
