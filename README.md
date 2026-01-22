# Stable Diffusion XAI App

This project makes generative AI more transparent and interpretable by integrating Explainable AI (XAI) techniques directly into a Stable Diffusion workflow.

Explainable AI (XAI) focuses on making complex machine learning systems understandable to humans, addressing transparency, trust, and interpretability challenges often associated with deep learning models. XAI helps users and developers understand how and why an AI system produces a particular output by exposing internal representations and decision logic.

## Features

- Text prompt input for Stable Diffusion image generation
- Gradient-based and perturbation-based visualization of internal representations
- Embedding inspection (for example, text-to-image alignment)
- Comparison of perturbed outputs to illustrate the influence of prompt changes
- Modular architecture for integrating additional explainability methods

## Installation

1. Clone the repository

```bash
git clone https://github.com/philiph930/Stable-Diffusion-XAI-App.git
cd Stable-Diffusion-XAI-App
```

3. Install dependencies
```bash
pip install -r requirements.txt
``` 

5. Create a .env file and add your Stable Diffusion API key
```bash 
STABLE_DIFFUSION_API_KEY=<your_key_here>
```

Alternatively, configure a local Stable Diffusion model path.

## Running the App

Start the application by running  
```bash
python app.py
```

Then open your browser and navigate to  
http://localhost:5000
