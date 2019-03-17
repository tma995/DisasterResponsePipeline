# Disaster Response Pipeline Project

### Overview:
Analyze messages data collected by <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a>, build a classification model and web services to help classify new messages for disaster response organizations.

### Installations:
1. Run following command for pre-required python libraries.
* `pip install -r requirements.txt`

### File Description
    .
    ├── app     
    │   ├── run.py                           # File to run app
    │   └── templates   
    │       ├── go.html                     
    │       └── master.html                   
    ├── data                   
    │   ├── disaster_categories.csv          # Categories data
    │   ├── disaster_messages.csv            # Messages data
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py              # ML training        
    ├── README.md    
    ├── misc                   
    │   ├── ETL Pipeline Preparation.ipynb 
    │   └── ML Pipeline Preparation.ipynb 
    └── requirements.txt                     # Python libraries

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
