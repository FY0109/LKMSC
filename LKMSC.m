function [L,output]=LKMSC(X,d,k,lambda,options,view)

if isfield(options,'maxiter') maxiter=options.maxiter;else maxiter=500;end
if isfield(options,'iter_D') iter_D=options.iter_D;else iter_D=5;end
if isfield(options,'tol') tol=options.tol;else tol=1e-4;end
if isfield(options,'init_type') init_type=options.init_type;else init_type='k-means';end
if isfield(options,'nrep_kmeans') nrep_kmeans=options.nrep_kmeans;else nrep_kmeans=100;end
if isfield(options,'obj_all') obj_all=options.obj_all;else obj_all=0;end

alpha=zeros(1,view);
[~,n]=size(X{1});
for i = 1:view
   alpha(i)=1/view;
   [m,~]=size(X{i});
   D{i}=initial_D(X{i},m,d,k,init_type,nrep_kmeans);
end


C=initial_C(X,D,alpha,view);
i=0;
while i<=maxiter
    i=i+1;
    
    if length(find(sum(abs(C))==0))>=n*0.1
        lambda=lambda/2;
        disp(['Too large lambda! Restart with lambda=' num2str(lambda) ' ......'])
        for i = 1:view
           [m,n]=size(X{i});
           D{i}=initial_D(X{i},m,d,k,init_type,nrep_kmeans);
        end
        C=initial_C(X,D,alpha,view);
        i=1;
    end
    C_new=update_C_Jacobi(X,D,C,d,k,lambda,alpha,view);
    D_new=D;
    for v=1:view
        D_new{v}=update_D(X{v},D{v},C_new,iter_D);
    end
  
    C=C_new;
    D=D_new;
    M = zeros(view,1);
    for iv = 1:view
        M(iv) = norm( X{iv}-D{iv}*C,"fro");
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    loss_temp=0;
for l=1:view
    loss_temp=loss_temp+0.5*alpha(l)^2*norm(X{l}-D_new{l}*C_new,'fro')^2;
end
loss(i)=loss_temp+lambda*sum(sum(reshape(C_new,d,k*n).^2).^0.5);
 end

%disp('Asign clusters by C')
for i=1:k
    Y(i,:)=sum((C((i-1)*d+1:i*d,:)).^2);
end
[~,L]=max(Y);


output.D=D;
output.C=C;
output.loss=loss;
end
%%
function D=initial_D(X,m,d,k,init_type,nrep_km)

X=X(:,randperm(size(X,2),min(size(X,2),50000)));

switch init_type
    case 'random'
        D=randn(m,d*k);
    case 'k-means'
        %disp(['Initializing D by k-means algorithm (' num2str(nrep_km) ' replicates)...'])
        [id,C,~,dist]=kmeans(X',k,'Distance','cosine','Replicates',nrep_km);
        [~,idx]=sort(dist,'ascend');
        for i=1:k
            temp=[X(:,idx(1:d,i))];
            if m<d
                D(:,(i-1)*d+1:i*d)=[temp];
            else
                if issparse(temp)
                    temp=full(temp);
                    [U,~,~]=svd(temp,'econ');
                else
                    [U,~,~]=svd(temp,'econ');
                end
                D(:,(i-1)*d+1:i*d)=U(:,1:d);
            end
        end
end
ld=sum(D.^2).^0.5;
idx=find(ld>1);
D(:,idx)=D(:,idx)./repmat(ld(idx),m,1);
end

% D=D./repmat(sum(D.^2).^0.5,m,1);
%%
function C=initial_C(X,D,alpha,view)
[~,n]=size(X{1});
[~,d]=size(D{1});
C=zeros(d,n);
for i=1:view
    C=C+inv(alpha(i)^2*D{i}'*D{i}+eye(d)*1e-5)*alpha(i)^2*D{i}'*X{i};
end
end
%%
function C_new=update_C_Jacobi(X,D,C,d,k,lambda,alpha,view)
tau=0;
C_new=C;
gC=zeros(size(C));
for v=1:view
    gC=gC+alpha(v)^2*(-D{v}')*(X{v}-D{v}*C);
    tau=tau+1.0*normest(alpha(v)*D{v})^2;
end
for j=1:k
%     tau=1.01*normest(D(:,(j-1)*d+1:j*d))^2;
    temp=C((j-1)*d+1:j*d,:)-gC((j-1)*d+1:j*d,:)/tau;
    C_new((j-1)*d+1:j*d,:)=solve_L21(temp,lambda/tau);
end
end

%%
function D_new=update_D(X,D,C_new,iter_D)
m=size(D,1);
D_t=D;
XC=X*C_new';
CC=C_new*C_new';
tau=1.0*normest(CC);
for j=1:iter_D
    gD=-XC+D_t*CC;
    D_t=D_t-gD/tau;
    ld=sum(D_t.^2).^0.5;
    idx=find(ld>1);
    D_t(:,idx)=D_t(:,idx)./repmat(ld(idx),m,1);
%     D_t=D_t./repmat(sum(D_t.^2).^0.5,m,1);
    if norm(gD/tau,'fro')/norm(D_t,'fro')<1e-3
        break
    end  
end
D_new=D_t;
end
%%
function X=solve_L21(X,thr)
L=(sum(X.^2)).^0.5;
Lc=max(0,L-thr)./L;
X=X.*repmat(Lc,size(X,1),1);
X(:,find(L==0))=0;
end