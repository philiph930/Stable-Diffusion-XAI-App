import urllib.request
from PIL import Image
import numpy as np
from numpy import array
import requests
import jpype
import asposecells
import json
import math
import torch
import sys
import re

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from operator import itemgetter

from itertools import chain
import accelerate
import pickle
import validators
import clip

import sklearn
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, pairwise_distances
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE

from functools import partial
import copy

import seaborn as sns
import matplotlib.pyplot as plt

from io import BytesIO


'''
LimeBase class
'''

class LimeBase(object):
    
    """
    Class for learning a locally linear sparse model from perturbed data.
    """

    def __init__(self, kernel_width=0.25, kernel=None, verbose=False, random_state=None):
        """
        Init function
        """

        if kernel is None:

            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.segments = None

    def _data_labels_text(self, model_inputs, label, black_box_img, classifier_fn, words, stop_indices, num_samples, batch_size, distance_metric, unk_id, pad_id):
        
        print("Starting perturbed sample processing")
        # Obtaining the clip model and preprocessor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        sample_num = 0
        
        word_ids = model_inputs[0]
              
        if not isinstance(word_ids, np.ndarray):
            word_ids = word_ids.numpy()
            
        ori_shape = word_ids.shape
        word_ids = word_ids.reshape((np.prod(ori_shape), ))

        if pad_id is None:
            n_features = len(word_ids)
        else:
            pad_locs = np.where(word_ids == pad_id)[0]
            n_features = word_ids.shape[-1] if len(pad_locs) == 0 else min(pad_locs)
        data = self.random_state.randint(0, 2, (num_samples + 1) * n_features) \
            .reshape((num_samples + 1, n_features))
        data[0, :] = 1
        samples = []
        data_set = []
        for row in data[1:]:
            temp = copy.deepcopy(word_ids)
            zeros = np.where(row == 0)[0]
            for z in zeros:
                temp[z] = unk_id

            samples.append(tuple(temp.reshape(ori_shape).tolist()))
            
            if len(samples) == batch_size:
                pred_inputs = (np.array(samples), ) + tuple(
                    [np.repeat(inp, batch_size, axis=0) for inp in model_inputs[1:]])
                
                classifier_fn(*pred_inputs, labels=label, prompt_words=words, perturbed_sample_num=sample_num)
                
                sample_num = sample_num + 10
                
                file = open('image_embeddings', 'rb')

                while True:
                    try:
                        temp = pickle.load(file)
                        data_set.append(temp)
                    except EOFError:
                        # End of file reached, break out of the loop
                        break
                
                samples = []

        print("samples array generated")

        if len(samples) > 0:
            pred_inputs = (np.array(samples), ) + tuple(
                [np.repeat(inp, len(samples), axis=0) for inp in model_inputs[1:]])
            
            classifier_fn(*pred_inputs, labels=label, prompt_words=words, perturbed_sample_num=sample_num)

            file = open('image_embeddings', 'rb')

            while True:
                try:
                    temp = pickle.load(file)
                    data_set.append(temp)
                except EOFError:
                    # End of file reached, break out of the loop
                    break

        # close the file
        file.close()

        print("Sample image encodings generated")
        
        black_box = preprocess(black_box_img).unsqueeze(0).to(device)

        with torch.no_grad():
            black_box_features = model.encode_image(black_box)
    
        black_box_features = black_box_features.numpy()

        black_box_features = list(chain.from_iterable(black_box_features))
        
        data_set.insert(0, black_box_features)
        
        
        samples = data.tolist()

        sample_num = 0
        stopword_num = 0

        for sample in data:
        
            for stopword in stop_indices:
                del(samples[sample_num][stopword - stopword_num])
                stopword_num = stopword_num + 1
        
            stopword_num = 0
            sample_num = sample_num + 1
        
        samples = np.array(samples)
        distances = sklearn.metrics.pairwise_distances(samples, samples[0].reshape(1, -1), metric=distance_metric).ravel()
            
        print("Samples, image encodings, distances returned")
        return samples, data_set, distances


'''
Helper Functions
'''

def classifier_fn(*args, labels, prompt_words, perturbed_sample_num) :
    
    #folder path that will store the perturbed image samples for data collection/evaluation
    folder_path = 'perturbed_samples/'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # open a file, where you want to store the data
    file = open('image_embeddings', 'wb')
    
    text = ""
    for i in range(len(args[0])):
        print("perturbed sample " + str(i) + " image generating starting...")
        for j in range(len(prompt_words)):
            if args[0][i][j] != 0:
                text = text + (labels[j]) + ' '
        
        '''
        while True:
            r = requests.post("https://api.deepai.org/api/stable-diffusion", data={
                'text': text, 'grid_size': '1'},
                headers={'api-key': '4a74fd5e-a8f6-49cf-90e2-de7d76d79f3b'})
        
            perturbed_image = r.json()
        
            # Construct the full file path
            perturbed_sample_file_path = folder_path + 'perturbed_image' + str(perturbed_sample_num) + '.png'
        
            image_url = perturbed_image.get("output_url")
        
            if not validators.url(image_url):
                continue
                
            else:
                break
        
        # Download the image from the URL
        urllib.request.urlretrieve(image_url, perturbed_sample_file_path)

        perturbed_sample_png = Image.open(perturbed_sample_file_path)
        '''

        while True: 
            # Set your Dezgo API key and endpoint
            dezgo_api_key = 'DEZGO-AF03075A00DF42F1F0E06ACBC15FB0F83653B04FCC8E101F1D1C4FB8597CD0673C623D8C'
            dezgo_api_endpoint = 'https://api.dezgo.com/text2image'

            # Set the prompt for the image
            prompt = text

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
                perturbed_sample_png = Image.open(image_data)

                break

            else:
                continue
        
        perturbed_im = preprocess(perturbed_sample_png).unsqueeze(0).to(device)

        with torch.no_grad():
            perturbed_sample_features = model.encode_image(perturbed_im)
    
        perturbed_sample_features = perturbed_sample_features.numpy()

        perturbed_sample_features = list(chain.from_iterable(perturbed_sample_features))

        text = ""
        
        perturbed_sample_num = perturbed_sample_num + 1
    
        # dump information to that file
        pickle.dump(perturbed_sample_features, file)
    
    # close the file
    file.close()
    
    return file


def n_most_important_features(n, samples, mean_coefficient_scores, total_words, neg_prompt):

    important_words = {}
    n = int(n)

    if n > len(samples[0]):
        print("Error: value specified is greater than number of words!")
        sys.exit()

    largest_coefficient_scores = sorted(abs(mean_coefficient_scores), reverse=True)[:n]
    counter = 0

    for i in range(n):
        k = 0
        for j in range(len(samples[0])):
            if largest_coefficient_scores[i] == abs(mean_coefficient_scores[j]) and counter < n:  
                if total_words[j] not in important_words:
                    if(mean_coefficient_scores[j] < 0):
                        important_words[total_words[j] + ' (neg prompt replacement)'] = mean_coefficient_scores[j]
                        k += 1
                    else:
                        important_words[total_words[j]] = mean_coefficient_scores[j] 
                    counter = counter + 1

    return important_words


def normalized_coefficient_scaling(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)

    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(scores)
    
    normalized_scores = (scores - min_val) / (max_val - min_val)
    return normalized_scores

'''
Main
'''

def main(prompt, int_prompt_labels_tensor, label_dict, black_box_image, num_features, words, stop_word_indices, cluster_centers, total_words, neg_prompt, replacement_words):

    print('Generating samples...')

    '''
    # Load pre-trained Word2Vec model
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    # dictionary that maps integer to its string value
    label_dict = {}

    # list to store integer labels
    int_prompt_labels = []

    #stop words that will be removed from text prompt
    stop_words = set(stopwords.words('english'))
    
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
    sentence = prompt
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


    while True: 
        # Set your Dezgo API key and endpoint
        dezgo_api_key = 'DEZGO-AF03075A00DF42F1F0E06ACBC15FB0F83653B04FCC8E101F1D1C4FB8597CD0673C623D8C'
        dezgo_api_endpoint = 'https://api.dezgo.com/text2image'

        # Set the prompt for the image
        prompt = sentence

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
        print('yes')
        
        if response.status_code == 200:

            print('success')
            # Decode the image data from the response
            image_data = BytesIO(response.content)

            # Open and display the image using Pillow
            black_box_image = Image.open(image_data)

            # Save the image to a file
            black_box_image.save('static/image.png')

            break

        else:
            continue
    
    '''

    ''' 
       
        #Request black box image from Stable Diffusion API and save as .png file
        r = requests.post(
            "https://api.deepai.org/api/stable-diffusion",
            data={
                'text': sentence, 'grid_size': '1'
            },
            headers={'api-key': '4a74fd5e-a8f6-49cf-90e2-de7d76d79f3b'}
        )

        black_box_image = r.json()

        black_box_image_url = black_box_image.get("output_url")

        if not validators.url(black_box_image_url):
            continue

        else:
            break

    urllib.request.urlretrieve(black_box_image_url, "image.png")

    black_box_image = Image.open('image.png')
    '''
    
    # Create an instance of LimeBase, which allows us to obtain representations of all perturbed text samples,
    # reduced embeddings of all perturbed images, and distances between each image and the original black box image
    lime_base = LimeBase(kernel_width=0.25, kernel=None, verbose=False, random_state=None)

    num_perturbed_images = 4

    samples, labels, distances = lime_base._data_labels_text(int_prompt_labels_tensor, label_dict, black_box_image, classifier_fn, words, stop_word_indices, num_samples=num_perturbed_images,
                                                         batch_size=10, distance_metric='euclidean',
                                                         unk_id=0, pad_id=None)
    
    
    # Create reduced embeddings list for all perturbed samples
    all_sample_reduced_embeddings = TSNE(n_components=2, learning_rate='auto', perplexity=3, random_state=42).fit_transform(np.array(labels))
    
    
    # For each image embedding, compare to cluster centroids via cosine distance
    # and use the distances to each centroid as the new sample representation
    cosine_distance_sample_representations = pairwise_distances(all_sample_reduced_embeddings, cluster_centers, metric='cosine')
    
    
    # Create MultiOutput Ridge Regressor model based on text samples, image embeddings, and weights calculated from distances
    weights = lime_base.kernel_fn(distances)
    model_regressor = MultiOutputRegressor(Ridge(alpha=1e-21, fit_intercept=True, random_state=lime_base.random_state))

    model_regressor.fit(samples, cosine_distance_sample_representations, sample_weight=weights)

    prediction_score = model_regressor.score(samples, cosine_distance_sample_representations, sample_weight=weights)
    
    # Calculate word importance mean score for each word we consider in the text prompt, stored in the list mean_coefficient_scores
    mean_coefficient_scores = []

    # Access the list of base estimators
    base_estimators = model_regressor.estimators_

    # Initialize a list to store the coefficients for each target variable
    coefficients = []

    # Iterate through the base estimators and retrieve the coefficients
    for estimator in base_estimators:
        if hasattr(estimator, 'coef_'):
            coefficients.append(estimator.coef_.tolist())

    for feature_index in range(len(samples[0])):

        feature_coefficients = []
        for n_target in range(17):
            # Extract the coefficients for the specific feature across all target variables
            feature_coefficients.append(coefficients[n_target][feature_index])

        abs_feature_coefficients = [abs(x) for x in feature_coefficients]

        # Measure the importance of the feature using a summary statistic (e.g., mean or sum)
        importance_score = np.mean(abs_feature_coefficients)

        mean_coefficient_scores.append(importance_score)
    
    mean_coefficient_scores = np.array(mean_coefficient_scores)
    
    # Normalize the scores
    normalized_coefficient_scores = normalized_coefficient_scaling(mean_coefficient_scores)
    
    for i in range(len(total_words)):
        total_words[i] = total_words[i].replace('.', '').replace(',', '').replace('!', '')
    
    for i in range(len(samples[0])):
        if total_words[i] in replacement_words:
            print(total_words[i])
            normalized_coefficient_scores[i] = 0 - normalized_coefficient_scores[i]

    print(normalized_coefficient_scores)
                
    # Retrieve certain number (user-input) of most important words and their importance scores
    word_importance_ordered = n_most_important_features(num_features, samples, normalized_coefficient_scores, total_words, neg_prompt)
    
    
    # Get the feature names (assuming you have them)
    feature_names = []

    for word in word_importance_ordered:
        feature_names.append(word)

    largest_coefficient_scores = word_importance_ordered.values()

    '''
    # Create a bar plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(range(len(feature_names)), largest_coefficient_scores)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Values')
    plt.title('Ridge Regression Coefficients')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()

    # Show the plot
    plt.show()
    '''
    
    print("result importances done")
    return word_importance_ordered
    #black_box_image_url
    
if __name__ == "__main__":
    word_importances = main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])
    print(word_importances)
