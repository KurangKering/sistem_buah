
[rDf, cDf] = size(df_uji);
[rCenters, cCenters, pCenters] = size(output_nilai_a_baru);

output_uji_layer1 = zeros(rCenters, cCenters, rDf);
for i=1:rDf
    for j=1:cCenters
       for k=1:rCenters
           output_uji_layer1(k,j,i) = gaussmf( df_uji(i,j), [output_nilai_a_baru(k,j,i), output_nilai_c_baru(k,j,i) ] );
       end
    end
end