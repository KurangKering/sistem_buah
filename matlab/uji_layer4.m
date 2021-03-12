[rDf, cDf] = size(df_uji);
[rOutput3, cOutput3] =  size(output_uji_layer3);
[rConse, cConse] = size(conse_epoch1);

nilai_f_uji = zeros(rDf, cOutput3);
output_uji_layer4 = zeros(rDf, cOutput3);
for i=1:rDf
    for j=1:cOutput3
        f = 0;
        for k=1:cConse
            if k < cConse
                f = f + (df(i,k) * conse_epoch1(j,k));
            else
                f = f + conse_epoch1(j,k);
            end
        end
        nilai_f_uji(i,j) = f;
        output_uji_layer4(i,j) = output_uji_layer3(i,j) * f;
    end
    
end