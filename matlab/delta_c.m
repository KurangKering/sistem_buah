

[rOutputErrorA, cOutputErrorA, pOutputErrorA] = size(output_errorc);

matrix_centers =  repmat(centers,[1,1,pOutputErrorA]);
output_delta_c = zeros(rOutputErrorA, cOutputErrorA, pOutputErrorA);
learning_rate = 0.001;
for i=1:pOutputErrorA
    for j=1:cOutputErrorA
        x = df(i,j);
       for k=1:rOutputErrorA
           center = matrix_centers(k,j,i);
           output_delta_c(k,j,i) = learning_rate * output_errorc(k,j,i) * x;
       end
    end
end
