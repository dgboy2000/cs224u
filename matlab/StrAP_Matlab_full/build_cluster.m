function cluster = build_cluster(idx,B,true_label)
%%%
%%% build the clustering model after initialization
%%%
%%%	idx:	the clustering indicator
%%%	B:	the dataset
%%%	true_label: the true label of data B
%%%
[uidx,n]=uniq_my(idx);
cluster.n=n; clear n
cluster.ex=B(uidx,:);
cluster.ex_idx=uidx;
cluster.meanstd=[];
cluster.label=true_label(uidx);
cluster.label_all={};
cluster.touch=repmat(size(B,1),length(cluster.n),1);

for k=1:length(cluster.n)
    elem=find(idx==uidx(k));
    tem1=repmat(cluster.ex(k,:),length(elem),1) - B(elem,:);
    tempd=diag(tem1*tem1');
    miu2=sum(tempd.^0.5);    %%% the sum of distance between elem and ex_k
    sigma2=sum(tempd);    %%% the sum of squared distance between elem and ex_k
    if cluster.n(k)>1
       miu=miu2/(cluster.n(k)-1);
    else
       miu=0;
    end
   
    if cluster.n(k)>2
                                                                         %                           1            
          sigma=((sigma2-(miu^2)*(cluster.n(k)-1))/(cluster.n(k)-2))^0.5;        %%% sigma = -------------- (sigma2 - cluster.n(k) * (miu^2) ) 
                                                                         %             cluster.n(k)-1                      
    else
        sigma=0;
%  
    end
    cluster.meanstd=[cluster.meanstd;[miu sigma miu2 sigma2]];

    if isnumeric(true_label)
        [ll_u f]=uniq_my(true_label(elem));
        cluster.label_all(k)={[ll_u f]};
    else
      [ll_u f]=uniq_my_cell(true_label(elem));
      cluster.label_all(k)={[ll_u num2cell(f)]};
    end
end

cluster.label_all=cluster.label_all';

%%% here, cluster.n DOES count the exemplar itself 
%%%       miu is the average of distances from all elem to exemplar, NOT counting "0" the self-distance of exempalr
%%%       sigma also does not count the self-distance, therefore, it is smaller than std((-S(elem,uidx(k))).^0.5) 