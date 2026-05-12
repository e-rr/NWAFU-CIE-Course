function [lx, ps, evs] = mynaive_bayest(tctors, tlasses, nsses, k)
    lx = zeros(nsses, size(tctors,2));  
    ps = zeros(nsses, 1);
    evs = zeros(size(tctors,2), 1); 
    for class=1:nsses
       fm = tctors(find(tlasses == (class-1)), :);
        part_1=sum(fm,1) + k;
        part_2=size(fm,1) + k * size(tctors,2);
        lis = (sum(fm,1) + k) ./ (size(fm,1) + k * size(tctors,2)); 
        lx(class, :) = lis;
        ps(class) = (size(fm,1) + k) / (size(tctors,1) + k*nsses); 
    end
    evs = ( (sum(tctors,1)+k) ./ (size(tctors,1)+k*2) )'; 
end