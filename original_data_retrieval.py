from PIL import Image
import re
import requests
from io import BytesIO
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from numpy import array
import torch

import sys


def main(prompt, cleaned_prompt): 

    print("Starting XAI Program")
    # Load pre-trained Word2Vec model
    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # dictionary that maps integer to its string value
    label_dict = {}

    # list to store integer labels
    int_prompt_labels = []

    #stop words that will be removed from text prompt
    print("Loading stopwords")
    stop_words = set(stopwords.words('english'))
    print("Stopwords loaded")
        
    cluster_centers = [
        array([-101.98082, 0.55857223], dtype=np.float32),
        array([-91.24885, 18.987246], dtype=np.float32),
        array([ 79.9255, -49.931133], dtype=np.float32),
        array([-38.12549, -47.220474], dtype=np.float32),
        array([ 14.379628, -49.469666], dtype=np.float32),
        array([ -8.439352, -57.749672], dtype=np.float32),
        array([62.245167, 35.25249 ], dtype=np.float32),
        array([60.398975, 10.248576], dtype=np.float32),
        array([ 49.366524, -16.382105], dtype=np.float32),
        array([16.571861, 21.42112 ], dtype=np.float32),
        array([-14.10396, 53.029278], dtype=np.float32),
        array([11.728146, 54.34886 ], dtype=np.float32),
        array([-26.777708,  -8.429125], dtype=np.float32),
        array([-34.922127, 18.214806], dtype=np.float32),
        array([26.09534, 1.6612635], dtype=np.float32),
        array([ 27.87239 , -30.990784], dtype=np.float32),
        array([0.6234783, -12.418384], dtype=np.float32)
    ]


    # Tokenize sentence into words
    sentence = cleaned_prompt
    sentence = sentence.lower()
    sentence_without_punctuation = re.sub(r'[^\w\s]', '', sentence)
    words = sentence_without_punctuation.split()

    # Use integers to represent each word in the int_prompt_labels list and create label dictionary mapping integers to words
    for i in range(len(words)):
        label_dict[i] = words[i]
        int_prompt_labels.append(i + 1)

    int_prompt_labels_tensor = (torch.tensor(int_prompt_labels),)

    total_words = re.sub(r'[^\w\s]', '', sentence)
    total_words = sentence.split()

    print("Converting sentence to label array")
    #Remove stop words from text prompt
    index = 0
    stop_word_indices = []
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word in stop_words:
            stop_word_indices.append(index)

        index = index + 1

    n = 0
    for i in range(len(stop_word_indices)):
        del(total_words[stop_word_indices[i] - n])
        n = n + 1

    for i in range(len(total_words)):
        if total_words.count(total_words[i]) > 1:
            count = 1
            for j in range(len(total_words)):
                if total_words[j] == total_words[i]:
                    total_words[j] = total_words[j] + "(" + str(count) + ")"
                    count = count + 1

    print("Stop words removed")

    while True: 
        print("Generating black box image")
        # Set your Dezgo API key and endpoint
        dezgo_api_key = 'DEZGO-AF03075A00DF42F1F0E06ACBC15FB0F83653B04FCC8E101F1D1C4FB8597CD0673C623D8C'
        dezgo_api_endpoint = 'https://api.dezgo.com/text2image'

        # Set the prompt for the image
        prompt = prompt.lower()

        # Set the headers with the API key
        headers = {
            'X-Dezgo-Key': dezgo_api_key,
        }

        # Set the data payload with the prompt
        data = {
            'prompt': prompt,
        }

        # Make the API request
        response = requests.post(dezgo_api_endpoint, headers=headers, data=data)
            
        if response.status_code == 200:

            # Decode the image data from the response
            image_data = BytesIO(response.content)

            # Open and display the image using Pillow
            black_box_image = Image.open(image_data)

            # Save the image to a file
            black_box_image.save('static/image.png')

            break

        else:
            continue

    print('black box image done')
    return int_prompt_labels_tensor, label_dict, black_box_image, words, stop_word_indices, cluster_centers, total_words

if __name__ == "__main__":
    int_prompt_labels_tensor, label_dict, black_box_image, words, stop_word_indices = main(sys.argv[1], sys.argv[2])