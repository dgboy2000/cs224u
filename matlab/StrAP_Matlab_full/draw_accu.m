function  [accu,err,outlier]=draw_accu(label_given_ll,restart)
%%%  evaluate the clustering quality 
%%%  INPUT:
%%%	label_given_ll:  label given list (-1 means in Reservoir, 0 means error, 1 means correct)
%%%	restart:	the restart time step
%%%  OUTPUT:
%%%	accu:		clustering accuracy along time
%%%	err:		clustering error rate along time
%%%	outlier:	percentage of outliers along time
N=length(label_given_ll);
accu=zeros(N,1);
err=zeros(N,1);
outlier=zeros(N,1);
correct=0;
error=0;
out=0;
for i=1:N
    if (label_given_ll(i)==1)
        correct=correct+1;
    elseif (label_given_ll(i)==0)
        error=error+1;
    else
        out=out+1;
    end
    accu(i)=correct/i;
    err(i)=error/i;
%     accu(i)=correct/(correct+error);
%     err(i)=error/(correct+error);
    outlier(i)=out/i;
end
accu=100*accu;
err=100*err;
outlier=100*outlier;
%%%%%%%%%%%%%%%%%%%%%%%

if nargin <2
 
 figure;plot(accu)
 xlabel('time step','Fontsize',12)
 ylabel('Accuracy (%)','Fontsize',12)
 set(gca,'fontsize',12')
 
 figure;plot(err)
 xlabel('time step','Fontsize',12)
 ylabel('Error rate (%)','Fontsize',12)
 set(gca,'fontsize',12')
else

 figure;plot(accu)
 hold on
 for i=1:length(restart)-1
     plot(restart(i),accu(restart(i)),'r*','MarkerSize',12)
 end
 xlabel('time step','Fontsize',12)
 ylabel('Accuracy (%)','Fontsize',12)
 set(gca,'fontsize',12')
 hl=legend('Accuracy','Restart point');
 set(hl,'Fontsize',12,'Location','Best')
 
 figure;plot(err)
 hold on
 for i=1:length(restart)-1
     plot(restart(i),err(restart(i)),'r*','MarkerSize',12)
 end
 xlabel('time step','Fontsize',12)
 ylabel('Error rate (%)','Fontsize',12)
 set(gca,'fontsize',12')
 hl=legend('Error rate','Restart point');
 set(hl,'Fontsize',12,'Location','Best')
end
 