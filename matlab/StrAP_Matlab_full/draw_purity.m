function  purity=draw_purity(N_res)
%%% evaluate the clustering quality in terms of clustering purity
%%% INPUT:	N_res:	the number result files	
name=['res' num2str(1)];
load(name);

ll=model.initial.label_all;
purity=[];
ssum=[]; mmean_sstd=[]; nn=[];
pi=0; err=0; ss=0;
temp=zeros(length(ll),1);
for i=1:size(ll,1)
    if isnumeric(ll{i})
	lll=ll{i}(:,2);
    else
        lll=cell2mat(ll{i}(:,2));
    end	
    pi=pi+max(lll)/sum(lll);
    err=err+sum(lll)-max(lll);
    ss=ss+sum(lll);   
    temp(i)=max(lll)/sum(lll);
end
purity=[purity; pi/i];
ssum=[ssum;ss];
nn=[nn;length(ll)];
mmean_sstd=[mmean_sstd;[mean(temp) std(temp)]];

for jj=1:N_res
name=['res' num2str(jj)];
load(name);

for R=1:length(restart)
    fieldname=['a_restart' num2str(R)];
    ll=getfield(model,fieldname);
    ll=ll.label_all;
    pi=0;
    err=0;
    ss=0;
    temp=zeros(length(ll),1);
    for i=1:size(ll,1)
	if isnumeric(ll{i})
           lll=ll{i}(:,2);
	else
           lll=cell2mat(ll{i}(:,2));
        end
        pi=pi+max(lll)/sum(lll);
        err=err+sum(lll)-max(lll);
        ss=ss+sum(lll); 
         temp(i)=max(lll)/sum(lll);
    end
    purity=[purity; pi/i];  
    ssum=[ssum;ss];
    nn=[nn;length(ll)];
    mmean_sstd=[mmean_sstd;[mean(temp) std(temp)]];
end

end

purity=100*purity;

%%%%%% purity1 == mmean_sstd(:,1)

figure;
     [AX,H1,H2] =plotyy([1:length(purity)], purity,[1:length(nn)],nn,'plot');
     set(get(AX(1),'Ylabel'),'String','Averaged purity of each cluster (%)','Fontsize',14) 
     set(get(AX(2),'Ylabel'),'String','Number of clusters','Fontsize',14)
     set(AX(1),'YLim',[min(purity)-20 100])
     set(AX(2),'YLim',[0 max(nn)*3])
     set(AX(2),'YTick',[0:50:max(nn)*2])
     set(AX(1),'YTick',[min(purity)-20-mod(min(purity)-20,5):5:100])
     set(AX(1),'FontSize',16,'fontname','times')
     set(AX(2),'FontSize',16,'fontname','times')
     
     set(AX(1),'XLim',[0 length(purity)+1])
     set(AX(2),'XLim',[0 length(purity)+1])
     set(AX(2),'XTick',[0:50:length(purity)+1])
     
     xlabel('Restarts','Fontsize',14) 
     set(H1,'LineStyle','-','Marker','*')
     set(H2,'LineStyle','--','Marker','.')
     h_legend=legend('Number of clusters','Averaged purity of each cluster');
     set(h_legend,'FontSize',14,'Location','Best');
    
     
     

 
