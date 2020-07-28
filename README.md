# Spoken-Language-Indentification-Using-GRU-and-LSTM
Classification of spoken language into English, Hindi or Mandarin - EE599 NLP project

## File Description
1. preprocessing.py : extract MFCC features and preprocess data. 
2. traning_model.py : train to classify the sequence and also classify in real time (every 25ms).
3. test_streaming_model.py : test real time classification (every 10ms). 

## Dependencies
1. Python 3.7  
2. tensorflow 2.1  
3. tensorflow-gpu 2.1  
4. librosa  

To install the complete list of dependencies, run:  
```
pip install -r requirements.txt
```

## Running the files:

### test_streaming_model.py:
The file can be run by giving the folder name containing the test files as a command line argument. Please make sure all the test files are in a single folder. Make sure to include /* at the end of the path and that the entire path is in single quotes.  
Example usage:  
```
python test_streaming_model.py '/home/ubuntu/test_folder/*'
```
### Note:
1. The audio data is sampled at 16kHz and MFCC features are generated using 25 msec frames (10msec for testing) with a 10 msec skip.
2. Silence is same in any language. Since silence is ignored, and the model might predict the silence randomly. The average of predictions over a timeframe has to be considered for a hard decision.
