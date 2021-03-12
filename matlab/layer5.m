[rOutput4, cOutput4] =  size(output_layer4);
output_layer5 = zeros(rOutput4,1);
for i=1:rOutput4
    output_layer5(i) = sum(output_layer4(i,:));
end