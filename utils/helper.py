import os
import pandas
from django.templatetags.static import static

def get_second_value(list_of_tuples, first_value):
	return next((y for x,y in list_of_tuples if x == first_value), None)

def get_url_citra(kelas):
    url = 'citra-kelas'
    image_filename = ['malaccensis.png', 'microcarpa.png']


    if (kelas == 0):
        url = os.path.join(url, image_filename[0])
    elif (kelas == 1):
        url = os.path.join(url, image_filename[1])
    else:
        url = None

    return url_citra(url)

def url_citra(url):
    return '<img width="100%" src="{}" alt="">'.format(static(url))
