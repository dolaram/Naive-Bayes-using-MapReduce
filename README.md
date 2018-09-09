# Naive-Bayes-using-MapReduce
Implementation of Naive Bayes Classifier using MapReduce on  Hadoop
Naive Bayes implementation was done using MapReduce and Local Machine implementation, the data set is DBPedia. The following are brief descriptions of the project files:

# Naive_Bayes_classifier.py
The python script implements the Naive Bayes classifier and calculate computaion time and a
accuracy on train, devel and test dataset on Local Machine.

# Mapreduce_Naive_Bayes.py 
The python script implements the Naive Bayes algorithm on the train, test and dev dataset and records the corresponding accuracies along with time taken to train the algorithm. The dictionary is prepared on hadoop mapreduce platform.

# mapper.py 
The mapper python script is used to mapping using hadoop streaimng for generating the (label, word, 1) stream output.

# reducer.py 
The reducer python script is used to mapping using hadoop streaimng for generating the (label, word, count) stream output.

# log.txt
Log file containing hadoop log records.

# Useful Material
# 1. Lemmatization
    https://pythonprogramming.net/lemmatizing-nltk-tutorial/
# 2. Removing Punctuation 
    https://www.programiz.com/python-programming/examples/remove-punctuation
# 3. Removing the Stop words
    https://pythonspot.com/nltk-stop-words/
# 4. Naive Bayes Classifier
    https://pdfs.semanticscholar.org/a397/eb310921897ef8a140668b623de618da7606.pdf
  
