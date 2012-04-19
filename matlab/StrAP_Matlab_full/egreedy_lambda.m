function current_lambda = egreedy_lambda(table_action ,eps)


if (rand < eps)
    current_lambda = ceil(rand() * size(table_action, 1));

else
    
%     stat = table_action(:,2) ./ table_action(:,3);
    stat = table_action(:,2) ./ (table_action(:,3)+1);
    [mx, idx] = min(stat);
    current_lambda = idx;
end


end
