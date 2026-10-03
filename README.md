# Sleep Health & Lifestyle Analytics System

## Project Overview

This project analyzes sleep health and lifestyle data using Python and Pandas. It provides a simple web interface to view dataset summaries and explore a machine-learning model that predicts a sleep-disorder category from selected inputs.

This project is for educational purposes only and is not a medical diagnostic tool.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Flask
* HTML
* CSS
* Machine Learning

## Main Features

* Displays basic dataset statistics.
* Summarizes sleep and lifestyle information.
* Preprocesses numerical and categorical features.
* Trains a Random Forest classification model.
* Displays model accuracy on a test dataset.
* Provides a web form for sample predictions.

## Project Structure

* `app.py` — Flask application and model logic
* `sleep_health.csv` — Sleep health dataset
* `templates/index.html` — Main webpage
* `static/style.css` — Webpage styling

## How to Run

1. Install Python.

2. Install the required libraries:

   `py -m pip install flask pandas scikit-learn`

3. Place the dataset in the project folder with the expected filename.

4. Run the application:

   `py app.py`

5. Open `http://127.0.0.1:5000` in your browser.

## Learning Outcomes

* Data loading and exploration using Pandas
* Handling numerical and categorical data
* Basic machine-learning model training and evaluation
* Building a simple Flask web application
* Connecting a model to an HTML form

## Disclaimer

Predictions are educational examples and must not be used to diagnose or treat any medical condition.
