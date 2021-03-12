[rDf, cDf] = size(df);
[rOutput3, cOutput3] =  size(output_layer3);
[rConse, cConse] = size(conse);

nilai_f = zeros(rDf, cOutput3);
output_layer4 = zeros(rDf, cOutput3);
for i=1:rDf
    for j=1:cOutput3
        f = 0;
        for k=1:cConse
            if k < cConse
                f = f + (df(i,k) * conse(j,k));
            else
                f = f + conse(j,k);
            end
        end
        nilai_f(i,j) = f;
        output_layer4(i,j) = output_layer3(i,j) * f;
    end
    
end