# text_classifier-svm
TEXT CLASSIFIER FOR UNSTRUCTURED SOCIAL MEDIA DATA

The numpy library can be installed as followed in Linux system as-

$ sudo apt install python3-numpy

$ pip3 install numpy 

OR
$ sudo apt install python3-numpy

The pandas library can be installed as follows-

$sudo apt-get install python-pip

$sudo pip install numpy

$sudo pip install pandas

The sklearn library can be installed as follows-

$sudo install -U scikit-learn

Preprocessing of data is done by using the NLP techniques such as tokenization, stop word removal and stemming.
Tokenization is done withe the help of CountVectorizer. Snowball Stemmer takes the tokenized TfidfTransformed data.
Stop word removal is done with the help of the class StemmedCountVectorizer. 
These techniques are integrated with the choosen ML algorithm (LinearSVC) using Pipeline.
