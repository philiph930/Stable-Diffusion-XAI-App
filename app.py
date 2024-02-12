from flask import Flask, render_template, request, redirect, url_for, session
import stablediffusion_xai_model

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from io import BytesIO
import base64

import numpy as np

import re
import requests
from PIL import Image


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def generateBarGraph(original_features_dict, adjusted_feature_values):
    print(session['word_importances'])
    words = list(original_features_dict.keys())
    original_weights = list(original_features_dict.values())

    print(f"{words=}")
    print(f"{original_weights=}")
    print(f"{adjusted_feature_values=}")

    x = np.arange(len(words))
    width = 0.35

    # Generate bar graph
    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, original_weights, width, label='Original Weights')
    rects2 = ax.bar(x + width/2, [value * weight for value, weight in zip(adjusted_feature_values, original_weights)], width, label='Adjusted Weights')

    ax.bar_label(rects1, padding=3, fmt='%1.2f')
    ax.bar_label(rects2, padding=3, fmt='%1.2f')

    #plt.bar(words, [bar_length * importance for bar_length, importance in zip(bar_lengths, importances)])
    ax.set_xlabel('Words', fontweight='bold')
    ax.set_ylabel('Weights (Importance)', fontweight='bold')
    ax.set_title('Top Feature Weights', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(words)
    ax.set_ylim(0.0, 2.2)
    ax.legend()


    # Save the plot to a BytesIO object
    img_bytesio = BytesIO()
    plt.savefig(img_bytesio, format='png')
    img_bytesio.seek(0)
    plt.close()

    # Convert the BytesIO object to base64 for embedding in HTML
    img_base64 = base64.b64encode(img_bytesio.getvalue()).decode('utf-8')

    return img_base64

def adjust(img_base64):
    if 'word_importances' in session:
        ordered_importances = dict(sorted(session['word_importances'].items(), key=lambda item: item[1], reverse=True))
        adjusted_feature_values = [float(request.form[f'{word}']) for word in request.form.keys() if word.startswith('feature_importance_')]
        adjusted_importances = {}

        prompt = session['prompt']
        words = list(ordered_importances.keys())

        i = 0
        for word in words:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
    
            # Replace the target word with "(word: weight)"
            prompt = pattern.sub(f"({word}: {adjusted_feature_values[i]})", prompt)

            adjusted_importances[word] = adjusted_feature_values[i]
            i = i + 1

        print(prompt)

        while True: 
            # Set your Dezgo API key and endpoint
            dezgo_api_key = 'DEZGO-AF03075A00DF42F1F0E06ACBC15FB0F83653B04FCC8E101F1D1C4FB8597CD0673C623D8C'
            dezgo_api_endpoint = 'https://api.dezgo.com/text2image'

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

        return render_template('barGraph.html', img_base64=img_base64, word_importances=adjusted_importances, prompt=prompt, img_url='static/image.png')


def update():
    if 'word_importances' in session:
        # Get a list of bar lengths from the form data
        ordered_importances = dict(sorted(session['word_importances'].items(), key=lambda item: item[1], reverse=True))
        adjusted_feature_values = [float(request.form[f'{word}']) for word in request.form.keys() if word.startswith('feature_importance_')]
        img_base64 = generateBarGraph(ordered_importances, adjusted_feature_values)

        adjusted_word_importances = {}

        i = 0
        for word in ordered_importances:
            adjusted_word_importances[word] = adjusted_feature_values[i]
            i = i + 1

        #return render_template('bar_graph.html', img_base64=img_base64, word_importances=adjusted_word_importances)
        #session['img_base64'] = img_base64
        return {'img_base64': img_base64} 
        #return render_template('bar_graph.html', img_base64=img_base64,word_importances=adjusted_importances, prompt=session['prompt'], img_url='static/image.png') 
    else:
        # Handle the case where word_importances is not a dictionary
        return "Unable to update"


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/main-page')
def mainpage():
    return render_template('mainpage.html');

@app.route('/your-url', methods=['GET', 'POST'])
def yourUrl():
    initial_bar_lengths = {}
    if request.method == 'POST':

        #if request.form.get('submit') == 'Update Graph':
            #return update()
        if request.form.get('promptSubmit') == 'Go':
            prompt = request.form['promptText']
            session['prompt'] = prompt
            num_features = request.form['featureNumber']
            word_importances = stablediffusion_xai_model.main(prompt, num_features)
            session['word_importances'] = word_importances
            
            default_ones = {}

            for word in word_importances:
                default_ones[word] = 1.0
            #session['black_box_img_url'] = black_box_image_url

            '''
            if initial_bar_lengths == {}:
                for word in word_importances.keys():
                    initial_bar_lengths[word] = 1
                
                bar_lengths = []
                for val in initial_bar_lengths.values():
                    bar_lengths.append(val)

                session['bar_lengths'] = bar_lengths

            else:
                # Get a list of bar lengths from the form data
                bar_lengths = [float(request.form[f'bar_length_{word}']) for word in request.form.keys() if word.startswith('bar_length_')]

            #return render_template('yourUrl.html', prompt=request.form['prompt'])
            '''

            # Check if word_importances is a dictionary
            if isinstance(word_importances, dict):
                img_base64 = generateBarGraph(word_importances, default_ones.values())
                #session['img_base64'] = img_base64
                return render_template('barGraph.html', img_base64=img_base64, word_importances=default_ones, prompt=prompt, img_url='static/image.png')
            else:
                # Handle the case where word_importances is not a dictionary
                return "Invalid data format. Expected a dictionary."
 
        elif request.form.get('submit') == 'Adjust Weights':
            ordered_importances = dict(sorted(session['word_importances'].items(), key=lambda item: item[1], reverse=True))
            adjusted_feature_values = [float(request.form[f'{word}']) for word in request.form.keys() if word.startswith('feature_importance_')]
            img_base64 = generateBarGraph(ordered_importances, adjusted_feature_values)
            return adjust(img_base64)
        
        else: 
            return update()
    else:
        return redirect(url_for('home'))