# Project : Natural Language Processing

## Sentiment Analysis with Deep Learning Models and Gradio App Deployment 

Sentiment analysis plays a vital role in understanding of public opinions and emotions. Dependency of human interactions via various social platforms necessitates the need for robust models to determine public sentiments, tones and attitudes on relevant subjects. The digital footprint left on diverse social networking sites is enormous and can be used to build and train Machine Learning Models for prediction/classification. In this classification project, I pretrained a deep learning model from Hugging Face Transformers library able to predict users text-inputs regarding covid vaccination. The text are either labelled Positive, Negative, or Neutral.

### Article
| Code      | Name        | Published Article |  Deployed App |
|-----------|-------------|:-------------:|------:|
| LP5  | Medium | https://medium.com/@benjaminkipkem/sentiment-analysis-gradio-app-fd0fc86cfd86 | [Best Article Machine learning](/) |
| LP5  | Hugging Face | https://huggingface.co/KAITANY/finetuned-roberta-base-sentiment | [Best Model](/) |
### Setup
- ### Setting up virtual Environment: 
Create a virtual environment to isolate current project dependencies from others. This isolates the required libraries of the project to avoid conflicts with other projects.  

You need to have [`Python3`](https://www.python.org/) on your system. To create a virtual environment using a terminal and a text editor of your choice, follow these steps: 

1. Open your terminal.
2. Navigate to your current working directory using the cd command. Example:

       cd /path/to/your/project/directory

3. Once in the desired directory, use the following commands to create and activate the virtual environment, upgrade pip, and create a    requirements.txt file.

- Windows:
        
        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt 
  

### Install the required packages remotely and locally. 
I installed necessary libraries for execution of the project. Data preprocessing and model finetuning was done remotely on Google Colab ['Google Colaboratory'](https://colab.research.google.com/), whereas App Development was done locally via my Vs Code editor. Google Colab is a free, cloud-based platform provided by Google that allows users to write and execute Python code collaboratively in a Jupyter Notebook-like environment. I chose Colab due to computational needs of working with large datasets.

### Data Understanding and EDA
Having integrated colab to my Google Drive, I was able to load both the training and test datasets from my drive into my colab Jupyter notebook. I used various pandas methods to gain understanding of the datasets stored in CSV files. There were few missing values in some of the columns that I opted to delete and visualization gave insight into distribution of data. 

### Data Preprocessing
Users text_inputs contained usernames, emojis, URLs, hashtags, and special characters. To avoid noise during finetuning of deep learning models and enhance their performance, I created a function to clean text_inputs by removing aforementioned characters. I transformed all text_inputs to lowercase to ensure all the text is treated in a consistent manner upon tokenization. I manually split the training set to have a training subset ( a dataset the model will learn on), and an evaluation subset ( a dataset the model will use to compute metric scores).

### Loading of Model
Hugging Face Transformers is an open-source library provided by Hugging Face that offers a collection of pre-trained models for various natural language processing (NLP) tasks. Hugging Face Transformers includes tokenizers that are tailored to specific pre-trained models and enable effective text processing. I went ahead to define the models to be used for tokenization and modelling. My choice models were 'cardiffnlp/twitter-roberta-base-sentiment-latest' and 'cardiffnlp/twitter-roberta-base-sentiment'. I also defined a function to be used by the “trainer” class to compute metrics on the evaluation dataset.

### Training of model & Evaluation
I configured various training parameters for fine-tuning the model using the Hugging Face Transformers library. These parameters are essential for controlling the training process, optimizing model performance, and managing the output of the fine-tuning process. The ‘Trainer’ class from Hugging Face Transformers is used for training and evaluation.  It takes various arguments including the model, training arguments (args), training dataset (train dataset), evaluation dataset (eval dataset), data collator, and a function (compute_metrics) for computing evaluation metrics. Training and evaluation is done concurrently by the "trainer" class and upon completion model performance scores are displayed.

### Push to Hub
Experimented with different training parameters and values till I attained a satisfactory model performance(prediction). I then pushed the finetuned model to the Hugging Face Model Hub for easy accessibility and sharing.

### App development
Hugging Face deep learning models can be integrated into friendly graphic user interface applications. Upon pushing best model to hub, I went ahead to create an interactive Gradio app that predicts the sentiments of users text-inputs related to Covid-19 vaccination. 

### Conclusion
This sentiment analysis project iterates the power of Hugging Face deep learning models to tackle various tasks including natural language processing. We have just seen finetuning and simple deployment of a deep learning model for Covid-19 sentiment analysis via Gradio. Outlined are steps to handle a similar project from preprocessing, training, and deployment. Depending on your task at hand, you can experiment with different Hugging Face pretrained models for finetuning and subsequent predictions/classifications.


### Author

##### Benjamin Kaitany