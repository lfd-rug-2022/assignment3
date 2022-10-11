
# Baseline LSTM

Trains a LSTM model using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.

## Usage,

*. One can simply run the file without any command line arguments because the script 
is set up with best settings as default values for arguments.

`python lfd_assignment3.py`

Running with the best hyperparams must give a macro F1-score of around 89/90.

Or if you want to change any of the arguments, please type

`python lfd_assignment3.py --help`