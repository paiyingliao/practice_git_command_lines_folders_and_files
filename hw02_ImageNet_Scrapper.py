# Created by Pai-Ying Liao (liao119, PUID: 0029934248) on Feb. 2nd, 2021 for ECE 695 DL HW2 task1
import json
import argparse
import requests
from PIL import Image
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
import os

parser = argparse.ArgumentParser(description='HW02 Task1')
parser.add_argument('--subclass_list', nargs='*', type=str, required=True)
parser.add_argument('--images_per_subclass', type=int, required=True)
parser.add_argument('--data_root', type=str, required=True)
parser.add_argument('--main_class', type=str, required=True)
parser.add_argument('--imagenet_info_json', type=str, required=True)
args, args_other = parser.parse_known_args()

with open(args.imagenet_info_json) as json_info:
    json_info = json.load(json_info)

ClassNameToID = {}
for key, value in json_info.items():
    ClassNameToID[value['class_name']] = key

the_list_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
if not os.path.exists(args.data_root):
    os.mkdir(args.data_root)
class_folder = args.data_root + "/" + args.main_class
if not os.path.exists(class_folder):
    os.mkdir(class_folder)

id = 0

def get_image(img_url, class_folder):
    if len(img_url) <= 1:
        print("url useless")
        return False
    try:
        img_resp = requests.get(img_url, timeout=1)
    except ConnectionError:
        print("Connection Error!")
        return False
    except ReadTimeout:
        print("Read Timeout!")
        return False
    except TooManyRedirects:
        print("Too Many Redirects!")
        return False
    except MissingSchema:
        print("Missing Schema!")
        return False
    except InvalidURL:
        print("Invalid URL!")
        return False

    if not 'content-type' in img_resp.headers:
        print("Missing Content!")
        return False
    if not 'image' in img_resp.headers['content-type']:
        print("The URL Doesn't Have Any Image!")
        return False
    if (len(img_resp.content) < 1000):
        print("Image Too Small (< 1 KB)!")
        return False

    img_name = img_url.split('/')[-1]
    img_name = img_name.split("?")[0]
    img_name = str(id) + "_" + img_name

    if (len(img_name) <= 1):
        print("Missing Image Name!")
        return False
    if not 'flickr' in img_url:
        print("Missing Non-flickr Image!")
        return False

    img_file_path = os.path.join(class_folder, img_name)
    # os.mkdir(img_file_path, args.main_class)

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)

    im = Image.open(img_file_path)
    if im.mode != "RGB":
        im = im.convert(mode="RGB")
    im_resized = im.resize((64, 64), Image.BOX)
    im_resized.save(img_file_path)
    return True

for subclass in args.subclass_list:
    full_url = the_list_url + ClassNameToID[subclass]
    resp = requests.get(full_url)
    urls = [url.decode('utf-8') for url in resp.content.splitlines()]

    count = 0

    for url in urls:
        if get_image(url, class_folder):
            count += 1
            id += 1
            if count >= args.images_per_subclass:
                print("Successfully downloaded and downsampled {} {} images!".format(args.images_per_subclass, subclass))
                break