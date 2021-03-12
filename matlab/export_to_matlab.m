arr = output_uji_layer1 
filename = 'output_uji_layer1'
for k=1:size(arr,3)
    xlswrite([filename '.xlsx'],arr(:,:,k),k);
    disp(k);
end