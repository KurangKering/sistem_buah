

[rCenters, cCenters, pCenters] = size(matrix_centers);
output_nilai_c_baru = zeros(rCenters,cCenters,pCenters);
for i=1:pCenters
    for j=1:cCenters
        for k=1:rCenters
            output_nilai_c_baru(k,j,i) = matrix_centers(k,j,i) + output_delta_c(k,j,i);
        end
    end
end