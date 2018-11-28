#pylint: disable=E1101
import io
import os
import os.path
import urllib.request

imglist = open("imgnet_fall11_urls.txt", "r", encoding="ISO-8859-15")
labels = open("imgnet_wnid.txt", "w")

for line in imglist.readlines():
    name = line.split("\t")[0]
    label = name.split("_")[0]
    labels.write(label)

    url = line.split("\t")[1]
    if os.path.isfile("{}.jpg".format(name)) == False: 
        os.system("wget -O {}.jpg {}".format(name, url))

labels.close()
imglist.close()


#urllib.request.urlretrieve("http://www.digimouth.com/news/media/2011/09/google-logo.jpg", "local-filename.jpg")