o1 = zeros(9,9,90)
for i=1:90
    for j=1:9
        for k=1:9
            o1(k,j,i) = gaussmf(df(i,j), [sigmas(1), centers(k,j)])
        end
    end
end