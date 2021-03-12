classes = df_with_class(:,end);
[rOutput5, cOutput5] =  size(output_layer5);
output_error5 = zeros(rOutput5,cOutput5);
for i=1:rOutput5
    output_error5(i) = -2*(classes(i) - output_layer5(i));
end
    
