
[rDf, cDf] = size(df);
[rCenters, cCenters] = size(centers);

[rOutputErrorA, cOutputErrorA, pOutputErrorA] = size(output_errora);

matrix_sigmas =  repmat(sigmas,[rOutputErrorA,1,pOutputErrorA]);
output_delta_a = zeros(rOutputErrorA, cOutputErrorA, pOutputErrorA);
learning_rate = 0.001;
for i=1:pOutputErrorA
    for j=1:cOutputErrorA
        x = df(i,j);
       for k=1:rOutputErrorA
           sigma = matrix_sigmas(k,j,i);
           output_delta_a(k,j,i) = learning_rate * output_errora(k,j,i) * x;
       end
    end
end
