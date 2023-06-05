clc
clear all
warning off
load('MSRC-v1.mat');
disp('Dateset：MSRC-v1')
Label=Y;
k=length(unique(Label));
data=X;
[~,view]=size(data);
ACC=[];
NMI=[];
Purity=[];
Fscore=[];
%% LKMSC

%%parameters

d=2;
p=10;

for rr=1:9
    rand('seed',rr);
for viewnum=1:view
    data{viewnum}=normalizeL2(data{viewnum}');
    Z=getZ(data{viewnum}',p,5);
    data{viewnum}=Z;
end
 [~,T] = aligned(data,0.1);
for vv=1:view
    data{vv}=T{vv}*data{vv};
end
opt.solver=0;
opt.maxiter=150;
opt.tol=1e-4;
opt.init_type='k-means';
opt.nrep_kmeans=5;
k=length(unique(Label));
lambda=0.15;

%%LKMSC
[L_kFSC,OUT]=LKMSC(data,d,k,lambda,opt,view);
data=X;
%%show res
res = Clustering8Measure(Label,L_kFSC);
fprintf(' P:%d \t d:%d\t ACC:%12.6f\t nmi:%12.6f\t Purity:%12.6f\t Fscore:%12.6f \t\n',[p d res(1) res(2) res(3) res(4) ]);
imagesc(abs(OUT.C))
leng=length(ACC)+1;
ACC(leng)=res(1);
NMI(leng)=res(2);
Purity(leng)=res(3);
Fscore(leng)=res(4);
end
ACC_av=mean(ACC);
NMI_av=mean(NMI);
Purity_av=mean(Purity);
Fscore_av=mean(Fscore);
disp("Avg");
fprintf('ACC:%12.6f\t nmi:%12.6f\t Purity:%12.6f\t Fscore:%12.6f \t\n',[ACC_av NMI_av Purity_av Fscore_av]);
ACC_s=std(ACC);
NMI_s=std(NMI);
Purity_s=std(Purity);
Fscore_s=std(Fscore);
disp("std");
fprintf('ACC:%12.6f\t nmi:%12.6f\t Purity:%12.6f\t Fscore:%12.6f \t\n',[ACC_s NMI_s Purity_s Fscore_s]);
