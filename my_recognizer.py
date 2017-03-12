import warnings
from asl_data import SinglesData
import numpy as np


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    # Loop over each word and then run all the models on that word, retaining
    # the log likelihood values for that word/model combination
    
    # get the feature sequences and lengths for each test word
    iTestWord = test_set.get_all_Xlengths()
    
    # Loop over each word, and the each model to find max likelihood model-word.
    for iWordIdx, iWordTuple in iTestWord.items():
        # initialize best guess word variable
        bestGuessWord = None
        maxLL = float('-inf')
        testWordDict = dict()  # Keep track of the probabilities for each model per test word
        testSequence = np.array(iWordTuple[0]) #hmm friendly format
        testLen = np.array(iWordTuple[1]) # hmm friendly format
        # Loop over each model and retain highest likelihood word
        for iModelWord, iModel in models.items():
            # Error trap for models that don't work for some reason.
            try:
                logL = iModel.score(testSequence, testLen)
            except:
                logL = float('-inf')  # set to default value for a failed model.
             
            # Store current model word results in dictionary    
            testWordDict[iModelWord] = logL    
            # retain highest likelihood word
            
            if logL > maxLL:
                maxLL=logL
                bestGuessWord = iModelWord

        # update guesses list with highest likelihood word
        guesses.append(bestGuessWord)
        probabilities.append(testWordDict)
            
    return probabilities, guesses