# N-Gram-Language-Model
# Put the java program and the datasets under the same directory

# The program takes at least 4 arguments
args[0]: the train set file name
args[1]: the file name containing data for which you need to make predictions
args[2]: the N for N-gram model
args[3]: the K for add-K Smoothing (No smoothing when K = 0)

Optional arguments:
args[4] to args[ngram + 3]: the parameters for linear interpolation
args[ngram + 4]: low frequency threshold for Named entity recognition
args[ngram + 5]: frequency threshold for unk

# Train the model without any smoothing
Unigram model
java HW1 brown.train.txt brown.dev.txt 1 0
Trigram model
java HW1 brown.train.txt brown.dev.txt 3 0

# Train the model with add-K smoothing
Unigram model with add-2 smoothing
java HW1 brown.train.txt brown.dev.txt 1 2

# Train the model with Linear Interpolation
Bigram model with parameters (lambda 1: 0.3, lambda 2: 0.7)
java HW1 brown.train.txt brown.dev.txt 2 0 0.3 0.7
Trigram model with parameters (lambda 1: 0.3, lambda 2: 0.4, lambda 3: 0.3)
java HW1 brown.train.txt brown.dev.txt 3 0 0.3 0.4 0.3

# Train the model with add-k smoothing and Linear Interpolation 
Bigram model with parameters (K: 3, lambda 1: 0.3, lambda 2: 0.7)
java HW1 brown.train.txt brown.dev.txt 2 3 0.3 0.7
