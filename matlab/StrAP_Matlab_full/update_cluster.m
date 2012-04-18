function newcluster = update_cluster(cluster,idx,new_set,idx_or,f_all,label_res1)
%%%	update the model
%%% Input:
%%%	cluster:	old cluster
%%%	idx:		clustering indicator (of old exemplars and elements in Reservoir)
%%%	new_set:	data of old exemplars and elements in Reservoir
%%%	idx_or:		indicator in original
%%%	f_all:		frequency of each data item
%%%	label_res1:	label of new_set
%%% Output:
%%%	newcluster:	new clustering model after update

        [uidx,n]=uniq_my(idx);
        Nc_old=length(cluster.n);
        N_all=length(idx);
        ssigma2_t=[cluster.meanstd(:,4);zeros(N_all-Nc_old,1)];
        mmiu2_t=[cluster.meanstd(:,3);zeros(N_all-Nc_old,1)];
        touch=[cluster.touch;idx_or((Nc_old+1):N_all)];
        label_all=[cluster.label;label_res1];
        for k=1:length(n)
            elem=find(idx==uidx(k));
                newcluster.ex(k,:)=new_set(uidx(k),:);
                newcluster.n(k)=sum(f_all(elem));
                newcluster.touch(k)=max(touch(elem));
                newcluster.ex_idx(k)=idx_or(uidx(k));
                newcluster.label(k)=label_all(uidx(k));
            f=newcluster.ex(k,:);
            e_old=new_set(elem,:);
            tem1=repmat(f,length(elem),1) - e_old;
            tempd=diag(tem1*tem1').*f_all(elem);       
            sigma2=sum(tempd)+sum(ssigma2_t(elem));   %%% the sum of squared distance between elem and ex_k
            tempd2=(diag(tem1*tem1').^0.5).*f_all(elem);       
            miu2= sum(tempd2)+mmiu2_t(uidx(k));                         %%% the sum of distance between elem and ex_k 
            if newcluster.n(k)>=2           %% if newcluster.n(k) > 1,   there will be a problem if n(k)=1.0002
                miu=miu2/(newcluster.n(k)-1);
            elseif newcluster.n(k) <= 1
                miu=0;
            else
                miu=miu2/newcluster.n(k);
            end
            if newcluster.n(k)>2                
%                                                                                                              1            
              sigma=abs((sigma2-(miu^2)*(newcluster.n(k)-1))/(newcluster.n(k)-2))^0.5;        %%% sigma = -------------- (sigma2 - cluster.n(k) * (miu^2) ) 
         %                                                                                                cluster.n(k)-1  
            else
                 sigma=0;
            end
            newcluster.meanstd(k,:)=[miu sigma miu2 sigma2];
            
%             %%% update the UI, CE, label_all
            elem_old=find(elem <= Nc_old);
            elem_res=find(elem > Nc_old);
            
            if isempty(elem_res)
                temp=cluster.label_all{elem(elem_old(1))};
                for j=2:length(elem_old)
                    temp=update_context_2(temp,cluster.label_all{elem(elem_old(j))});
                end
                newcluster.label_all(k)={temp};
            elseif isempty(elem_old)
                if isnumeric(label_res1)
                    [ucc, kcc]=uniq_my(label_res1(elem(elem_res)-Nc_old));
                    temp=[ucc kcc];
                else
                    [ucc, kcc]=uniq_my_cell(label_res1(elem(elem_res)-Nc_old));
                    temp=[ucc num2cell(kcc)];
                end
                
                newcluster.label_all(k)={temp};
            else
                if isnumeric(label_res1)
                    [ucc, kcc]=uniq_my(label_res1(elem(elem_res)-Nc_old));
                    temp1=[ucc kcc];
                else
                    [ucc, kcc]=uniq_my_cell(label_res1(elem(elem_res)-Nc_old));
                    temp1=[ucc num2cell(kcc)];
                end
                    
                temp=cluster.label_all{elem(elem_old(1))};
                for j=2:length(elem_old)
                    temp=update_context_2(temp,cluster.label_all{elem(elem_old(j))});
                end
                temp=update_context_2(temp,temp1);
                newcluster.label_all(k)={temp};
            end

                
         end

newcluster.n=newcluster.n';
newcluster.touch=double(newcluster.touch)';
newcluster.ex_idx=newcluster.ex_idx';
newcluster.label=newcluster.label';
newcluster.label_all=newcluster.label_all';

%%% here, cluster.n DOES count the exemplar itself
%%%       miu is the average of distances from all elem to exemplar, not counting "0" the self-distance of exempalr
%%%       sigma also does not count the self-distance, therefore, it is
%%%       smaller than std((-S(elem,uidx(k))).^0.5) 
