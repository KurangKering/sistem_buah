import os
import pandas


def get_second_value(list_of_tuples, first_value):
	return next((y for x,y in list_of_tuples if x == first_value), None)

def get_url_citra(kelas):
    image_name = 'citra-kelas'
    image_filename = ['malaccensis.png', 'microcarpa.png']


    if (kelas == 0):
        image_name = os.path.join(image_name, image_filename[0])
    elif (kelas == 1):
        image_name = os.path.join(image_name, image_filename[1])
    else:
        image_name = None

    return image_name
