import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, words: dict, hwords: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=None, verbose=False):
        self.words = words
        self.hwords = hwords
        self.sequences = words[this_word]
        self.X, self.lengths = hwords[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
 
        #
        # Need to loop over range of possible numbers of hidden states (outer loop)
        # and BIC computations.  Return model with lowest BIC.  This code 
        # is adapted from my implementation of the SelectorCV since some of the 
        # code logic is similar.
        #
        # Let's be aggressive on run-time warnings, such as those issued from the modeling section below.
        warnings.filterwarnings("error", category=RuntimeWarning) # Raise an exception.
        
        #initialize global BIC over search space
        minBIC = float('inf')
        # Initialize the bestModel to return
        bestModel = None
        numDataPoints = len(self.X)
        
        print("------")
        print("SelectorBIC: curWord = ", self.this_word)
        print("SelectorBIC: numDataPoints = ", numDataPoints)
        print("SelectorBIC: X:")
        print(self.X)
        print("SelectorBIC: lengths")
        print(self.lengths)
        
        # Loop over range of hidden nodes.           
        for iHidden in range(self.min_n_components, self.max_n_components+1):
            print("SelectorBIC: iHidden = ", iHidden)
        
            # Error trap for bad training or scoring cases.
            try:
                model = GaussianHMM(n_components=iHidden, covariance_type="diag", 
                                    random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                    
                logL = model.score(self.X, self.lengths)
            
                numParams = 1
                curBIC = -2.0 * logL + numParams * math.log10(numDataPoints)
                
            except: #If there are any issues with training or scoring, set BIC to inf so below test doesn't pass
                curBIC = float('inf')
                    
            # Save this model parameters if it has the highest avgLL so far
            if curBIC < minBIC:
                minBIC=curBIC
                bestModel = model  # choose the best model in group so far

        if bestModel == None:  # Return a default case if there was a problem
            return None
        else:
            # This is the final model with the lowest BIC.
            return bestModel


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        return None
#        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        
        #
        # Need to loop over range of possible numbers of hidden states (outer loop)
        # and cross-fold validations, computing the average likelihood for each number of hidden
        # states.  Return the HMM with the greatest average likelihood.
        #
        # Pattern for model call:  SelectorCV(sequences, Xlengths, word, 
        #            min_n_components=2, max_n_components=15, random_state = 14).select()
        #
        # Let's be aggressive on run-time warnings, such as those issued from the modeling section below.
        warnings.filterwarnings("error", category=RuntimeWarning) # Raise an exception.
        
        # The number of splits cannot be greater than number of samples and also must be >=2 (per KFold doc)
        # If len(self.length) is < 3, then let 2 be selected automatically, otherwise set to 3 splits.
        if len(self.lengths) < 2:
            return None  # self.base_model(self.n_constant)
        minSplit = min(len(self.lengths),3)
        split_method = KFold(n_splits=minSplit)
        
        #initialize global max log likelihood over search space
        maxLL = float('-inf')
        # Initialize the bestModel to return
        bestModel = None
                    
        for iHidden in range(self.min_n_components, self.max_n_components+1):
            # initialize the avg log likelihood for the current number of hidden nodes
            avgLL = 0.0 # this is an accumulator, so okay to set to zero
            validSplit = 0 # Keep track of number of splits that can be trained & scored.
            maxLocalLL = float('-inf') # best log likelihood for this number of hidden nodes
            bestLocalModel = None  # store the best model of the group for this number of hidden nodes.
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
        
                # Use concatenation formalism from asl_data.py for hmm data format
                X_train_cat = []
                lengths_train = []
                
                # Create new arrays that extract the k-fold split indices for train and test hold-outs
                for iExtract in cv_train_idx:
                    X_train_cat = X_train_cat+self.sequences[iExtract]
                    lengths_train.append(self.lengths[iExtract])

                X_train = np.array(X_train_cat)
                
                # Now do similar setup for the holdout(test)/ cross-validation sequences.               
                X_holdout_cat = []
                lengths_holdout = []
                
                for iExtract in cv_test_idx:
                    X_holdout_cat = X_holdout_cat+self.sequences[iExtract]
                    lengths_holdout.append(self.lengths[iExtract])

                X_holdout = np.array(X_holdout_cat)
                  
                # Train on extracted  subset of sequences for k-fold
                # Error trap for bad training or scoring cases.
                try:
                    model = GaussianHMM(n_components=iHidden, covariance_type="diag", 
                                        random_state=self.random_state, n_iter=1000).fit(X_train, lengths_train)
                    # Want to score on the hold-out samples
                    logL = model.score(X_holdout, lengths_holdout)
                    validSplit = validSplit+1 # Valid split found (trained & scored)
                    # Keep track of the best model within this hidden node group.
                    if logL > maxLocalLL:
                        maxLocalLL = logL
                        bestLocalModel = model
                except: #If there are any issues with training or scoring, set log likelihood to zero
                    logL = 0
                    
                avgLL = avgLL + logL

            # Compute average log likelihood over the number of valid k-fold splits.
            if validSplit > 0:
                avgLL = avgLL/(1.0*validSplit)
            else:
                avgLL = float('-inf') # skip this model entirely (won't pass next if statement for updating model)
            
            # Save this model parameters if it has the highest avgLL so far
            if avgLL > maxLL:
                maxLL=avgLL
                bestModel = bestLocalModel  # choose the best model in group so far (arbitrarily)

        
        if bestModel == None:  # Return a default case if there was a problem
            return None
        else:
            # This is the best model in the best group.
            return bestModel


