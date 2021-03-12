[rOutput3, cOutput3] =  size(output_layer3);

output_error3 = zeros(rOutput3,cOutput3);
for i=1:rOutput3
    for j=1:cOutput3
        output_error3(i,j) = output_error4(i,j) * nilai_f(i,j);
    end
end
    
