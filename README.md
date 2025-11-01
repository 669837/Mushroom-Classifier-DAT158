# Mushroom Classifier
A simple Streamlit web app that classifies mushrooms as edible or poisonous using a Random Forest model trained on the UCI Mushroom dataset.

## Live Demo
You can try the app here: https://mushroom-classifier-dat158.streamlit.app/

## Run the app locally
To start the application, open a terminal in the project directory and:
1) Install dependencies: `pip install -r requirements.txt`
2) Run the app: `streamlit run src/app.py`

A pre-trained model is included at `models/mushroom_model.pkl` for convenience.  
To retrain/replace it, run: `python src/train_mushroom_model.py`  
This will overwrite `models/mushroom_model.pkl`.
