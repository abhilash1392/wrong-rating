# Wrong Rating Detection 

## Description 
This is a web-based app which can tell the executives the whether the poor rating given by the customer matches the review. Sometimes it happens that customer writes a good review but gives a bad rating by mistake, or thinks that 3 star is a good rating. 

## Directory Structure 
The directory structure is defined below:

    1. input - Contains the training data 
    2. models - Saved models
    3. notebooks - jupyter notebooks used for exploratory data analysis and initial modelling.
    4. src -  scripts used for building the app.

### src
The project has 4 major parts.

    1. cleanText.py - This contains the code which is used to clean the text data
    2. train.py - This contains the code which uses SentimentIntensityAnalyzer and RandomForestClassifier to train a pipeline using input file.
    3. test.py - This contains the predict function used for checking the rating
    4. app.py - This contains the code which is used to build the web app using flask app.

## Model Description
Following steps are taken to build the model.

    1. Loading the data 
    2. Selecting the required features of 'Text' and 'Star'
    3. Cleaning textual data, removing emojis and dropping na rows.
    4. Using Lexicon-based model to classify the text into positive,negative,neutral and compound.
    5. If rating is less than 3 and we check the probaility of review text being postive. If that is the case we identify the review as wrong.
    6. Did this for all training examples.
    7. Next we build a pipeline which has two steps.
        1. Vectorizing the text using CountVectorizer
        2. RandomForestClassifier
    8. Fit the pipeline with features as Text and target as weather the rating is right or wrong.
    9. Save the pipeline and edited file.

## Running the project 

1. Ensure you have all the files. Run the following command in the terminal

    ```
    python src/train.py
    ```

2. Run app.py using below command to start FlaskAPI 
    
    ```
    python src/app.py
    ```

3. Navigate to the url.

4. Input the review and rating, Check if negative rating is correctly given.

5. To upload a whole file navigate to url upload_file

6. A csv file identifying wrong ratings would be downloaded.

7. To run the code in colab use notebook/wrongTextRating.ipynb

## Limitations

The model cannot identify scarcasm.
Example:
Input: ("This app is amazing at sucking.",1)
This would identify this as wrong rating.

The model cannot deal with spelling mistakes tried using TextBlob but that lead to even bad results as "This is awsome" became "His is some"
