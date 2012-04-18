function model_online_report(ssii)
%%% showing the online clustering results as bars with label

name=['res' num2str(1)];
load(name);

ex_label=model.initial.label_all;

%%%%%%%%%%%% figure
figure;
for si=1:ssii
  name=['res' num2str(si)];
  load(name);

 for i=1:length(restart)
     name1=['b_restart' num2str(i)];
     r=i+(si-1)*30;   %%% the r-th restart	
     pie=pie_chart.(name1);
     pie=pie./sum(pie)*100;
     pie_selected=[1 find(pie(2:end)>1)+1];
     bar(pie(pie_selected))
     mp=max(pie(pie_selected));
     axis([0 length(pie_selected)+1 0 ceil(mp*1.1)+1])
     title(['the assignment of datas between restart ' num2str(r-1) ' and restart ' num2str(r)], 'fontsize',16)
     xlabel('Clusters','Fontsize',14)
     ylabel('Percentage of data assigned (%)','Fontsize',14)     
     hold on
     text(0.5,pie(1)+mp*0.1,'Reservoir')
     for j=2:length(pie_selected)
	tem=ex_label{pie_selected(j-1)};
	[m1,m2]=max(tem(:,2));
	tem2=num2str([tem(m2,1); 100*m1/sum(tem(:,2))]);
	tem2=[tem2 [' ';'%']];
         text(j-0.2,pie(pie_selected(j))+mp*0.1,tem2);
     end
     hold off

      name2=['a_restart' num2str(i)];
      ex_label=model.(name2).label_all;
      pause
 end

end

