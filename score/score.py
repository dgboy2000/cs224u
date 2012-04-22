import numpy

class KappaScore:
    """A single Kappa score on a single dataset."""
    def __init__(self, rater_a, rater_b, min_rating=None, max_rating=None):
        assert(len(rater_a)==len(rater_b))
        self.rater_a = rater_a
        self.rater_b = rater_b
        self.min_rating = min_rating
        self.max_rating = max_rating
        if self.min_rating is None:
            self.min_rating = min(reduce(min, self.rater_a), reduce(min, self.rater_b))
        if self.max_rating is None:
            self.max_rating = max(reduce(max, self.rater_a), reduce(max, self.rater_b))
    def confusion_matrix(self):
        """
        Returns the confusion matrix between rater's ratings
        """
        num_ratings = self.max_rating - self.min_rating + 1
        conf_mat = [[0 for i in range(num_ratings)]
                    for j in range(num_ratings)]
        for a,b in zip(self.rater_a,self.rater_b):
            conf_mat[a-self.min_rating][b-self.min_rating] += 1
        return conf_mat

    def histogram(self, ratings):
        """
        Returns the counts of each type of rating that a rater made
        """
        num_ratings = self.max_rating - self.min_rating + 1
        hist_ratings = [0 for x in range(num_ratings)]
        for r in ratings:
            hist_ratings[r-self.min_rating] += 1
        return hist_ratings

    def quadratic_weighted_kappa(self):
        """
        Calculates the quadratic weighted kappa
        scoreQuadraticWeightedKappa calculates the quadratic weighted kappa
        value, which is a measure of inter-rater agreement between two raters
        that provide discrete numeric ratings.  Potential values range from -1  
        (representing complete disagreement) to 1 (representing complete
        agreement).  A kappa value of 0 is expected if all agreement is due to
        chance.
    
        scoreQuadraticWeightedKappa(rater_a, rater_b), where rater_a and rater_b
        each correspond to a list of integer ratings.  These lists must have the
        same length.
    
        The ratings should be integers, and it is assumed that they contain
        the complete range of possible ratings.
   
        score_quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
        is the minimum possible rating, and max_rating is the maximum possible
        rating
        """
        conf_mat = self.confusion_matrix()
        num_ratings = len(conf_mat)
        num_scored_items = float(len(self.rater_a))

        hist_rater_a = self.histogram(self.rater_a)
        hist_rater_b = self.histogram(self.rater_b)

        numerator = 0.0
        denominator = 0.0

        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i]*hist_rater_b[j]
                          / num_scored_items) 
                d = pow(i-j,2.0) / pow(num_ratings-1, 2.0)
                numerator += d*conf_mat[i][j] / num_scored_items
                denominator += d*expected_count / num_scored_items

        return 1.0 - numerator / denominator


class MeanKappaScore():
    """A mean weighted Kappa score across a few datasets."""
    def __init__(self, scores=[], weights=[]):
        self.scores = list(scores)
        self.weights = list(weights)
        
    def add(self, score, weight=1.0):
        assert isinstance(score, KappaScore)
        self.scores.append(score)
        self.weights.append(weight)
    
    def mean_quadratic_weighted_kappa(self, weights=None):
        """
        Calculates the mean of the quadratic
        weighted kappas after applying Fisher's r-to-z transform, which is
        approximately a variance-stabilizing transformation.  This
        transformation is undefined if one of the kappas is 1.0, so all kappa
        values are capped in the range (-0.999, 0.999).  The reverse
        transformation is then applied before returning the result.
    
        mean_quadratic_weighted_kappa(kappas), where kappas is a vector of
        kappa values

        mean_quadratic_weighted_kappa(kappas, weights), where weights is a vector
        of weights that is the same size as kappas.  Weights are applied in the
        z-space
        """
        kappas = numpy.array([score.quadratic_weighted_kappa() for score in self.scores], dtype=float)
        if weights is None:
            if self.weights:
                weights = numpy.asarray(self.weights)
            else:
                weights = numpy.ones(numpy.shape(kappas))
        weights = weights / numpy.mean(weights)

        # ensure that kappas are in the range [-.999, .999]
        kappas = numpy.array([min(x, .999) for x in kappas])
        kappas = numpy.array([max(x, -.999) for x in kappas])
    
        z = 0.5 * numpy.log( (1+kappas)/(1-kappas) ) * weights
        z = numpy.mean(z)
        kappa = (numpy.exp(2*z)-1) / (numpy.exp(2*z)+1)
        return kappa
        
        
        
        
        
        
        
        
        
        
        
        
        