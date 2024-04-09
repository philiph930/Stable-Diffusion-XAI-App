from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
import stablediffusion_xai_model
import original_data_retrieval

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from io import BytesIO
import base64

import numpy as np

import re
import requests
from PIL import Image

from threading import Thread
import asyncio

from openai import OpenAI
from dotenv import load_dotenv

import json

import time
import os

load_dotenv('API_KEY.env')
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
socketio = SocketIO(app, logger=True, engineio_logger=True)

global shared_data
shared_data = {"img": {}, "importances": {}}

client_word_importances = {}

async def generate_image_async(prompt, num_features, shared):

    cleaned_prompt = re.sub(r'[\\(\\):0-9.]', '', prompt)

    print("starting original data retrieval")
    int_prompt_labels_tensor, label_dict, black_box_image, words, stop_word_indices, cluster_centers, total_words = original_data_retrieval.main(prompt, cleaned_prompt)
    shared_data["img"] = black_box_image
    socketio.emit('update_image', json.dumps({'img_url': 'static/image.png'}))

    word_importances = stablediffusion_xai_model.main(cleaned_prompt, int_prompt_labels_tensor, label_dict, black_box_image, num_features, words, stop_word_indices, cluster_centers, total_words)
    shared_data["importances"] = word_importances
    socketio.emit('send_word_importances', word_importances)

    default_imp = {}

    for word in word_importances:
        default_imp[word] = 0.50

    if isinstance(word_importances, dict):
        img_base64 = generate_bar_graph(word_importances, default_imp.values())
        
    socketio.emit('graph-image', json.dumps({'img_base64': img_base64}))

    #return render_template('bar_graph.html', img_base64=img_base64, word_importances=default_ones, prompt=prompt, img_url='static/image.png')
    return word_importances

def start_async_task(prompt, num_features, shared):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_image_async(prompt, num_features, shared))

def generate_bar_graph(original_features_dict, adjusted_feature_values):
    #print(session['word_importances'])
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
    rects2 = ax.bar(x + width/2, [value * weight * 2 for value, weight in zip(adjusted_feature_values, original_weights)], width, label='Adjusted Weights')

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
    #if 'word_importances' in session:
    #ordered_importances = dict(sorted(session['word_importances'].items(), key=lambda item: item[1], reverse=True))
    ordered_importances = dict(sorted(shared_data['importances'].items(), key=lambda item: item[1], reverse=True))
    adjusted_feature_values = [float(request.form[f'{word}']) for word in request.form.keys() if word.startswith('feature_importance_')]
    adjusted_importances = {}

    prompt = session['prompt']
    num_features = session['num_features']
    words = list(ordered_importances.keys())

    i = 0
    for word in words:

        print(adjusted_feature_values[i])

        if adjusted_feature_values[i] != 0.50:

            numSigns = (adjusted_feature_values[i] - 0.50) / 0.125
            print(numSigns)
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)

            if numSigns > 0:
                adjustment = '+' * int(numSigns)
                
                # Replace the target word with "(word: weight)"
                prompt = pattern.sub(f"{word}" + adjustment, prompt)
                print(prompt)
            
            else:
                adjustment = '-' * int(abs(numSigns))
                print(adjustment)
                
                # Replace the target word with "(word: weight)"
                prompt = pattern.sub(f"{word}" + adjustment, prompt)
                print(prompt)

            adjusted_importances[word] = adjusted_feature_values[i]
        
        i = i + 1

    print(prompt)

    if request.form['negPrompt'] != '' and not request.form['negPrompt'].isspace():

        print('yes')
        print(request.form['negPrompt'].isspace())
        negativePrompt = request.form['negPrompt'].split(', ')
        negativePromptString =  ', '.join(negativePrompt)
        print(negativePromptString)
        print('done')

        response = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            temperature = 0.0,
            max_tokens = 3000,
            messages = [
                {"role": "system", "content": "You are a prompt generator that helps a user"},
                {"role": "user", "content": "Generate exactly: " + prompt + "with no" + negativePromptString}
            ]
        )

        prompt = response.choices[0].message.content
        print(prompt)

    '''
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

        #else:
            #continue 
    ''' 

    thread = Thread(target=start_async_task, args=(prompt, num_features, shared_data))
    thread.start()

    importances = {}
    for i in range(int(num_features)):
        importances['word ' + str(i)] = 1

    #return render_template('bar_graph.html', img_base64={}, word_importances=importances, prompt=prompt, img_url={})
    return render_template('bar_graph.html', img_base64=shared_data['importances'],word_importances=importances, prompt=prompt, img_url=shared_data['img'])

    #return render_template('bar_graph.html', img_base64=img_base64, word_importances=adjusted_importances, prompt=prompt, img_url='static/image.png')


def update():
    #print(session['word_importances'])

    #if 'word_importances' in session:
        # Get a list of bar lengths from the form data
    #ordered_importances = dict(sorted(session['word_importances'].items(), key=lambda item: item[1], reverse=True))
    ordered_importances = dict(sorted(shared_data['importances'].items(), key=lambda item: item[1], reverse=True))
    adjusted_feature_values = [float(request.form[f'{word}']) for word in request.form.keys() if word.startswith('feature_importance_')]
    img_base64 = generate_bar_graph(ordered_importances, adjusted_feature_values)

    adjusted_word_importances = {}

    i = 0
    for word in ordered_importances:
        adjusted_word_importances[word] = adjusted_feature_values[i]
        i = i + 1

        #return render_template('bar_graph.html', img_base64=img_base64, word_importances=adjusted_word_importances)
        #session['img_base64'] = img_base64
    return {'img_base64': img_base64, 'word_importances': adjusted_word_importances} 
        #return render_template('bar_graph.html', img_base64=img_base64,word_importances=adjusted_importances, prompt=session['prompt'], img_url='static/image.png') 
    #else:
        # Handle the case where word_importances is not a dictionary
    #return "Unable to update"
    
@socketio.on('my event')
def handle_my_custom_event(json):
    print('received json: ' + str(json))

@socketio.on('session_importances')
def handle_importances(importances):
    print('received importances: ' + str(importances))
    session['word_importances'] = importances
'''
@socketio.on('connect') 
def connect():
    socketio.emit('response', 'Received: ')
    print('received connect')
'''
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/your-url', methods=['GET', 'POST'])
def your_url():
    print(shared_data)
    initial_bar_lengths = {}
    if request.method == 'POST':

        #render_template('bar_graph.html', img_base64={}, word_importances={}, prompt={}, img_url={})
        #if request.form.get('submit') == 'Update Graph':
            #return update()
        
        '''
        if request.form.get('submit') == 'Render':
            prompt = session['prompt']
            ordered_importances = dict(sorted(shared_data['importances'].items(), key=lambda item: item[1], reverse=True))

            importances={}
            for key in ordered_importances.keys():
                importances[key] = 1

            img_base64 = generate_bar_graph(shared_data['importances'], importances.values())
                
            return render_template('bar_graph.html', img_base64=img_base64,word_importances=importances, prompt=prompt, img_url='static/image.png')
        '''
             
        if request.form.get('submit') == 'Generate':
            prompt = request.form['prompt']
            session['prompt'] = prompt
            num_features = request.form['number_features']
            session['num_features'] = num_features
            
            #generate_image_async(prompt, num_features)
            
            thread = Thread(target=start_async_task, args=(prompt, num_features, shared_data))
            thread.start()
            #asyncio.create_task(generate_image_async(prompt, num_features))

            '''
            int_prompt_labels_tensor, label_dict, black_box_image, words, stop_word_indices, cluster_centers, total_words = original_data_retrieval.main(prompt)
            socketio.emit('update_image', {'img_url': 'static/image.png'}, namespace='/your-url')
            word_importances = stablediffusion_xai_model.main(prompt, int_prompt_labels_tensor, label_dict, black_box_image, num_features, words, stop_word_indices, cluster_centers, total_words)
            session['word_importances'] = word_importances
            '''
            '''
            default_ones = {}

            for word in session['word_importances']:
                default_ones[word] = 1.0
            '''
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

            #return render_template('your_url.html', prompt=request.form['prompt'])
            '''

            # Check if word_importances is a dictionary
            
            #if isinstance(session['word_importances'], dict):
            '''
            img_base64 = generate_bar_graph(session['word_importances'], default_ones.values())
            '''
            #session['img_base64'] = img_base64
            #return render_template('bar_graph.html', img_base64=img_base64, word_importances=default_ones, prompt=prompt, img_url='static/image.png')
            importances = {}
            for i in range(int(num_features)):
                importances['word ' + str(i)] = 0.50

            #return render_template('bar_graph.html', img_base64={}, word_importances=importances, prompt=prompt, img_url={})
            return render_template('bar_graph.html', img_base64=shared_data['importances'],word_importances=importances, prompt=prompt, img_url=shared_data['img'])
            #else:
                # Handle the case where word_importances is not a dictionary
                #return "Invalid data format. Expected a dictionary."
 
        elif request.form.get('submit') == 'Adjust Weights':
            print(request.form)
            #ordered_importances = dict(sorted(session['word_importances'].items(), key=lambda item: item[1], reverse=True))
            ordered_importances = dict(sorted(shared_data['importances'].items(), key=lambda item: item[1], reverse=True))
            adjusted_feature_values = [float(request.form[f'{word}']) for word in request.form.keys() if word.startswith('feature_importance_')]
            img_base64 = generate_bar_graph(ordered_importances, adjusted_feature_values)
            return adjust(img_base64)
        
        else: 
            return update()
    else:
        return redirect(url_for('home'))
    
if __name__ == '__main__':
    socketio.run(app, debug=True)