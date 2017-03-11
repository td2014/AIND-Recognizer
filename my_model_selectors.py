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
        return None
#        raise NotImplementedError


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
        # model = SelectorCV(sequences, Xlengths, word, 
        #            min_n_components=2, max_n_components=15, random_state = 14).select()
        #
        print()
        print("-----")
        print("SelectorCV: self.this_word: ", self.this_word)
###        print("SelectorCV: X:")
###        print(self.X)
###        print("SelectorCV: self.lengths:")
###        print(self.lengths)
###        print("SelectorCV: length of self.lengths = ", len(self.lengths))
###        print("SelectorCV: self.min_n_components = ", self.min_n_components)
###        print("SelectorCV: self.max_n_components = ", self.max_n_components)
###        print()
        
        # The number of splits cannot be greater than number of samples and also must be >=2 (per KFold doc)
        # If len(self.length) is < 3, then let 2 be selected automatically, otherwise set to 3 splits.
        minSplit = min(len(self.lengths),3)
###        print("SelectorCV: minSplit = ", minSplit)
        split_method = KFold(n_splits=minSplit)
        
        #initialize max log likelihood over search space
        maxLL = float('-inf')
        # Initialize the bestModel to return
        bestModel = None
        best_num_components=3 #default size in case we don't find better model (due to errors etc.)
                    
        for iHidden in range(self.min_n_components, self.max_n_components+1):
###            print("SelectorCV: iHidden = ", iHidden)
            # initialize the avg log likelihood for the current number of hidden nodes
            avgLL = 0.0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
###                print("SelectorCV: Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
        
                # Use concatenation formalism from asl_data.py for hmm data format
                X_train_cat = []
                lengths_train = []
                
                # Create new arrays that extract the k-fold split indices for train and test hold-outs
                for iExtract in cv_train_idx:
###                    print("SelectorCV: iExtract (train) = ", iExtract)
                    X_train_cat = X_train_cat+self.sequences[iExtract]
                    lengths_train.append(self.lengths[iExtract])

                X_train = np.array(X_train_cat)
###                print("SelectorCV: X_train:")
###                print(X_train)
###                print("SelectorCV: lengths_train:")
###                print(lengths_train)
                
                # Now do similar setup for the holdout(test)/ cross-validation sequences.               
                X_holdout_cat = []
                lengths_holdout = []
                
                for iExtract in cv_test_idx:
###                    print("SelectorCV: iExtract (test) = ", iExtract)
                    X_holdout_cat = X_holdout_cat+self.sequences[iExtract]
                    lengths_holdout.append(self.lengths[iExtract])

                X_holdout = np.array(X_holdout_cat)
###                print("SelectorCV: X_holdout:")
###                print(X_holdout)
###                print("SelectorCV: lengths_holdout:")
###                print(lengths_holdout)
                  
                # Train on extracted  subset of sequences for k-fold
                model = GaussianHMM(n_components=iHidden, n_iter=1000).fit(X_train, lengths_train)
              
                # Want to score on the hold-out samples
                logL = model.score(X_holdout, lengths_holdout)
###                print("SelectorCV: (holdout) logL = {}".format(logL))
                avgLL = avgLL + logL

            # Compute average log likelihood over the number of k-fold splits. (either 2 or 3)
            avgLL = avgLL/(1.0*minSplit)
###            print("SelectorCV: avgLL = ", avgLL)
            
            # Save this model parameters if it has the highest avgLL so far
            if avgLL > maxLL:
                print("SelectorCV: Updating best model.")
                maxLL=avgLL
                bestModel = model
                best_num_components=iHidden
                print ("SelectorCV: maxLL = ", maxLL)
                print ("SelectorCV: best_num_components = ", best_num_components)
                print ("SelectorCV: bestModel = ", model)
            
            print()
        
        if bestModel == None:  # Return a default case if there was a problem
            return self.base_model(best_num_components)
        else:
            # This is the final model in the best group.
            # Could have picked another one (best fit in that group)
            # but this is probably adaquate, as long as it came from the 
            # best group of fits
            print("SelectorCV: Returning a best fit model.")
            return bestModel


