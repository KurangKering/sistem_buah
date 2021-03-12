

[rSigmas, cSigmas, pSigmas] = size(matrix_sigmas);
output_nilai_a_baru = zeros(rSigmas,cSigmas,pSigmas);
for i=1:pSigmas
    for j=1:cSigmas
        for k=1:rSigmas
            output_nilai_a_baru(k,j,i) = matrix_sigmas(k,j,i) + output_delta_a(k,j,i);
        end
    end
end