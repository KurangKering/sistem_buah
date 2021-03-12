[rOutput2, cOutput2] =  size(output_layer2);

output_error2 = zeros(rOutput2,cOutput2);
for i=1:rOutput2
    w_total = sum(output_layer2(i,:),2);
    for j=1:cOutput2
        e_4_2 = 0;
        de_3_2 = 0;
        for k=1:cOutput2
            if j==k
                de_3_2 = (w_total - output_layer2(i,k)) / (w_total^2);
            else
                de_3_2 = -(output_layer2(i,k) / (w_total^2));
            end
            
            e_4_2 = e_4_2 + (output_error3(i,k) * de_3_2);
        end
        output_error2(i,j) = e_4_2;
    end
        
end
    
