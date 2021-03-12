[rOutput4, cOutput4] =  size(output_uji_layer4);
output_uji_layer5 = zeros(rOutput4,1);
for i=1:rOutput4
    output_uji_layer5(i) = sum(output_uji_layer4(i,:));
end