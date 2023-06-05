function [B]=getZ(X,anchor,r)
    X(find(isnan(X)==1)) = 0;
    kmMaxIter = 5;
    kmNumRep = 1;
    [nSmp,~]=size(X);
    [~,marks]=litekmeans(X,anchor,'MaxIter',kmMaxIter,'Replicates',kmNumRep);

    D = EuDist2(X,marks,0);
    sigma = mean(mean(D));
    dump = zeros(nSmp,r);
    idx = dump;
    for i = 1:r
        [dump(:,i),idx(:,i)] = min(D,[],2);
        temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
        D(temp) = 1e100; 
    end
    
    dump = exp(-dump/(2*sigma^2));
    sumD = sum(dump,2);
    Gsdx = bsxfun(@rdivide,dump,sumD);
    Gidx = repmat([1:nSmp]',1,r);
    Gjdx = idx;
    Z=sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,anchor);
    B=Z';
end