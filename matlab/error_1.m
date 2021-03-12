[rOutput1, cOutput1, bOutput1] =  size(output_layer1);

output_error1 = size(output_layer1);
for i=1:bOutput1
   for j=1:cOutput1
      for k=1:rOutput1
          
          output_error1(k,j,i) = output_error2(i,k) * (prod(output_layer1(k,:,i))/ output_layer1(k,j,i)) ;
          
      end
   end
        
end
    
