# English_Word_Sentiment-Analysis
Character level sentiment analysis using CNN and LSTM
# Dependancies
* Used Python Version:3.7.0
* Install necessary modules with `sudo pip3 install -r requiremnets.txt` command.
# Model Training and Testing:
To train and test the model --> `python3 train_and_test.py`
# Model Parameters:
For LSTM:
  * Hidden_dimension = 50
  * Output_dimension = 3
  * Optimizer = Adadelta
  * Loss criterion = Binary Cross Entropy with LogitsLoss

For CNN
  * Output_dimension = 3
  * Dropout = 0.01
  * Optimizer = Adadelta
  * Loss criterion = Binary Cross Entropy with LogitsLoss
# Author:
Subhrajit Dey(@subro608)
