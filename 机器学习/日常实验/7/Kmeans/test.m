clc;
clear all;
close all;
load('MNIST.mat')
[Idx,Ctrs,SumD,D] = kmeans(X,10,'Replicates',4);
tmp = horzcat(Idx, Y);
coe = sortrows(tmp,1);
rs = zeros(10,1);
for i=1:10
    for j=1:10
        rs(i,j) = -length(find(coe(coe(:,1)==i,2)==j));
    end
end 
[ast,cost] = mumes(rs);
for i=1:10
    coe(coe(:,1)==i,3) = find(ast(i,:)==1);
end
acc = -cost/3000
NMI = nmi(coe(:,3)', coe(:,2)')
figure;
aa=coe(:,3);
bb=coe(:,2);
A=categorical(aa);
B=categorical(bb);
plotconfusion(A,B)