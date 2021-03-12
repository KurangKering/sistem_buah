
[rDf, cDf] = size(df);
[rCenters, cCenters] = size(centers);

output_errora = zeros(rCenters, cCenters, rDf);
for i=1:rDf
    for j=1:cCenters
        x = df(i,j);
        sigma = sigmas(j);
       for k=1:rCenters
           c = centers(k,j);
           output_errora(k,j,i) = output_error1(k,j,i) * (exp(-power((x-c)/sigma, 2.0)/2))*power(x-c, 2.0)/power(sigma, 3.0);
       end
    end
end