[rOutput, cOutput, pOutput]= size(output_uji_layer1);

output_uji_layer2 = zeros(pOutput, rOutput);
for i=1:pOutput
    for j=1:rOutput
        output_uji_layer2(i,j) = prod(output_uji_layer1(j,:,i));
    end
end