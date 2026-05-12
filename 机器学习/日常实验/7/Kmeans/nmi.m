function Mt = nmi(A, B)
if length(A) ~= length(B)
    error('length( A ) must == length( B)');
end
N = length(A);
Ad = unique(A);
KA = length(Ad);
Bid = unique(B);
KB = length(Bid);
Aur = double (repmat( A, KA, 1) == repmat( Ad', 1, N ));
Bur = double (repmat( B, KB, 1) == repmat( Bid', 1, N ));
ABur = Aur * Bur';
PA= sum(Aur') / N;
PB = sum(Bur') / N;
PAB = ABur / N;
Mx = PAB .* log(PAB ./(PA' * PB)+eps);
MI = sum(Mx(:));
HA = -sum(PA .* log(PA + eps),2);
HB= -sum(PB .* log(PB + eps),2);
Mt = MI / sqrt(HA*HB);