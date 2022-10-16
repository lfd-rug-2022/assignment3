
# Baseline LSTM

Trains a LSTM model using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.


## Data 

Data files are provided as train.txt, dev.txt, and test.txt

## Usage of lstm_baseline.py

*. One can simply run the file without any command line arguments because the script 
is set up with best settings as default values for arguments.

`python lstm_baseline.py`

Running with the best hyperparams must give a macro F1-score of around 89/90.

Or if you want to change any of the arguments, please type

`python lstm_baseline.py --help`

## Usage for bert_variants.py

Trains a BERT model variants using train & dev set. If test set is mentioned, accuracy/F1-score (Macro)
is calculated with an option of showing the confusion matrix.

To run this file,

1. One can simply run the file without any command line arguments because the script 
is set up with best settings as default values for arguments.

python bert_variants.py

Running with the best hyperparams must give a macro F1-score of around 89/90.

2. Or if you want to change any of the arguments, please type

python bert_variants.py --help

3. If you want to try any other bert variant, please type

python bert_variants.py --langmodel_name distilbert-base-uncased

Or for other models, try these:

microsoft/deberta-v3-base
xlnet-base-cased
roberta-base
bert-base-uncased
distilbert-base-uncased
albert-base-v2