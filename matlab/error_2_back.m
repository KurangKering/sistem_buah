[rOutput2, cOutput2] =  size(output_layer2);

output_error2 = zeros(rOutput2,cOutput2);
for i=1:rOutput2
    w_total = sum(output_layer2(i,:),2);
    subtract_err3 = output_error3(i, 1);
    
    for j=2:cOutput2
        subtract_err3 = subtract_err3 - output_error3(i, j);
    end
    for k=1:cOutput2
        output_error2(i,k) = ((w_total - output_layer2(i,k)) /  w_total^2) * subtract_err3;
    end
        
end
    
