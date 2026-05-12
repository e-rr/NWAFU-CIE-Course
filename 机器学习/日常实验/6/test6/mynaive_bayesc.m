function [pclasses, pors] = mynaive_bayesc(vs, prs, ls, evidence)
    n_classes = size(prs, 1);   
    n_vectors = size(vs, 1);    
    pclasses = zeros(n_vectors, 1);
    pors = zeros(n_vectors, n_classes);   
    for i=1:n_vectors
        vector_1 = find(vs(i, :)' == 1);   
        likelihood_v_1 = ls(:, vector_1);    
        vector_0 = find(vs(i,:)'==0);           
        likelihood_v_0=1-ls(:,vector_0);
        posterior =prod(likelihood_v_0,2) .*prod(likelihood_v_1,2) .* prs;  
        [max_val, class] = max(posterior);
        pclasses(i) = class-1;
        pors(i,:) = posterior';
    end
end
