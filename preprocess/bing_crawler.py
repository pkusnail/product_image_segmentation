import sys
sys.path.append("/usr/local/lib/python3.7/site-packages/")
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials

import os
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import argparse
import json
import time



search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
MAX_RESULTS = 2000
GROUP_SIZE = 50

subscription_key = "2e408a4dc18d4f3d927614a21ccc0ad8"

#####################

def save_img(url,out_dir):
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print("error status for ", url)
    ext = "."+url.split(".")[-1]
    ext = ext.split("!")[0]
    ext = ext.split("&")[0]
    ext = ext.split("?")[0]
    ext = ext.split("=")[0]
    pname = int(time.time()*1000)
    p = os.path.sep.join([out_dir, "{}{}".format(str(pname), ext)])
    #p = out_dir +"/" + str(pname) +"/"+ext
    print("saving  ", p)
    f = open(p, "wb")
    f.write(r.content)
    f.close()

####################
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True, help="search query to search Bing Image API for")
#ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
params  = {"q": args["query"], "license": "public", "imageType": "photo","offset": 0, "count": GROUP_SIZE}

output= "./"+str(args["query"])
os.mkdir(output)

response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
#total search result number
totalEstimatedMatches=search_results["totalEstimatedMatches"]
nextOffset=search_results["nextOffset"]
print("totalEstimatedMatches for keyword ", totalEstimatedMatches)

#all url msgs
values = search_results["value"]
for one in values:
    print(one["contentUrl"])
    try:
        save_img(one["contentUrl"], output)
    except Exception as e:
        print(e)


reqNum = min(totalEstimatedMatches, MAX_RESULTS)
while nextOffset < totalEstimatedMatches:
    print("[INFO] making request for group {}-{} of {}...".format(nextOffset, nextOffset + GROUP_SIZE, reqNum))
    params["offset"] = nextOffset
    search = requests.get(search_url, headers=headers, params=params)
    search.raise_for_status()
    search_results = search.json()
    vals = search_results["value"]
    nextOffset=search_results["nextOffset"]
    for one in vals:
        try:
            save_img(one["contentUrl"], output)
        except Exception as e:
            print(e)





