import string
import torch


def create_array(category):
    l = [0, 0, 0]
    arr = 1 - (category[0] + category[1])

    if arr > category[0] and arr > category[1]:
        l = [0, 0, 1]
    elif category[0] > arr and category[0] > category[1]:
        l = [1, 0, 0]
    elif category[1] > arr and category[1] > category[0]:
        l = [0, 1, 0]
    elif category[0] == category[1]:
        l = [0, 0, 0]
    elif category[0] == arr and arr > category[1]:
        l = [1, 0, 0]
    elif category[1] == arr and arr > category[0]:
        l = [0, 1, 0]
    return l


def letters():
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    return all_letters, n_letters


def preprocess_data():
    categories = {}
    with open("EnglishSentiWordNet_3.0.0_20130122.txt", "r") as f:
        lines = f.readlines()
    im_lines = lines[26:]
    with open("yourfile.txt", "w") as f:
        for line in im_lines:
            f.write(line)
    with open("yourfile.txt", "r") as f:
        lines = f.readlines()
        for f in lines[1:]:
            l = f.split()
            im_list = list(map(float, l[2:4]))
            try:
                a = create_array(im_list)

                categories[l[4].rstrip('#1')] = a

            except:
                print("Exception has occured")
    return categories
