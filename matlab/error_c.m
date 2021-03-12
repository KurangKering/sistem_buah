
[rDf, cDf] = size(df);
[rCenters, cCenters] = size(centers);

output_errorc = zeros(rCenters, cCenters, rDf);
for i=1:rDf
    for j=1:cCenters
        x = df(i,j);
        sigma = sigmas(j);
       for k=1:rCenters
           c = centers(k,j);
           output_errorc(k,j,i) = output_error1(k,j,i) * (exp(-power((x-c)/sigma, 2.0)/2)*(x-c))/(power(sigma, 2.0));
       end
    end
end