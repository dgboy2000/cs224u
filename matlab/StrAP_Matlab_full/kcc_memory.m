function [idx,dpsim]=kcc_memory(S,K,Ko,cm,nruns,maxits);
% k-centers with memory
% Input:
%   S         Similarity matrix
%   K         Number of clusters to find
%   Ko        Number of clusters in memory (previous model)
%   cm	      the proportion of centers from memory, K*cm number of initial centers are from memory(the previous model )
%   nruns     Number of runs to try, where each run is initialized randomly (default 1)
%   maxits    Maximum number of iterations (default 100)
% Ouput:
%   idx(i,j)    Index of the data point that data point i is assigned
%               to in the jth run. idx(i,j)=i indicates that point i
%               is an exemplar
%   dpsim(m,j)  Sum of similarities of data points to exemplars, after
%               iteration m in the jth run

if nargin<4 error('kcc:1','Too few input arguments');
elseif nargin==4 nruns=1; maxits=100;
elseif nargin==5 maxits=100;
elseif nargin>6 error('kcc:2','Too many input arguments');
end;
if length(K)==1 k=K; ui=0; else k=length(K); ui=1; end;

n=size(S,1); mod=0;
if Ko < (n/2)
   mod=1;Koc=floor(Ko*cm);
else
    Knc=floor((n-Ko)*cm);
end
 dpsim=zeros(maxits,nruns); idx=zeros(n,nruns);
for rep=1:nruns
    if ui 
        mu=K;
    elseif mod
        tmp1=randperm(Ko)'; mu=tmp1(1:Koc);
        tmp2=randperm(n-Ko)'; 
	if (k-Koc)>(n-Ko)
		mu=[mu;tmp2+Ko];
		k=length(mu);
	else
		mu=[mu;tmp2(1:(k-Koc))+Ko];
	end	
    else
       tmp1=randperm(Ko)'; 
       if (k-Knc) > Ko 
	       mu=tmp1;
	       k=length(mu)+Knc;
       else
	       mu=tmp1(1:(k-Knc));
       end
       tmp2=randperm(n-Ko)'; mu=[mu;tmp2(1:Knc)+Ko];
    end;
    i=0; dn=(i==maxits);
    while ~dn
        i=i+1; muold=mu; dpsim(i,rep)=0;
        [tmp cl]=max(S(:,mu),[],2); % Find class assignments
        cl(mu)=1:k; % Set assignments of exemplars to themselves
        for j=1:k % For each class, find new exemplar
            I=find(cl==j);
            [Scl ii]=max(sum(S(I,I),1));
            dpsim(i,rep)=dpsim(i,rep)+Scl(1);
            mu(j)=I(ii(1));
        end;
%         if i>1
%         cooldp=dpsim(i,rep)-dpsim(i-1,rep);
%         ccontinu=(cooldp>0);
%         end
%         if ((sum(sort(muold)==sort(mu))==k)&&~ccontinu)||(i==maxits) dn=1; end;
        if (sum(sort(muold)==sort(mu))==k)||(i==maxits) dn=1; end;
    end;
    idx(:,rep)=mu(cl); dpsim(i+1:end,rep)=dpsim(i,rep);
end;
