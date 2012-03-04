class Curve:
    """A curve for a grade distribution."""
    def __init__(self, scores, histogram=None, probs=None):
        if histogram is not None:
            self.set_curve_with_histogram(scores, histogram)
        elif probs is not None:
            self.set_curve_with_probs(scores, probs)
    def curve(self, score):
        """Return an integer grade for a numeric score"""
        grade = self.min_grade
        for cutoff_score in self.cutoff_scores:
            if score < cutoff_score:
                return grade
            grade += 1
        return self.max_grade
        
    def set_curve_with_histogram(self, scores, grade_counts):
        """Find the score cutoffs that separate each of the different possible grades."""
        possible_grades = grade_counts.keys()
        possible_grades.sort()
        last_grade = possible_grades[0]
        for grade in possible_grades[1:]:
            for i in range(last_grade + 1, grade):
                grade_counts[i] = 0
                #print "WARNING: did not specify count for grade %d; only saw these grades: %s" %(last_grade + 1, str(possible_grades))
            last_grade = grade
        self.min_grade = possible_grades[0]
        self.max_grade = possible_grades[-1]
        
        if len(scores) != sum(grade_counts.values()):
            raise Exception("ERROR: found %d scores and %d grades; must be same number" %(len(scores), sum(grade_counts.values())))
        if grade_counts[possible_grades[0]] * grade_counts[possible_grades[-1]] == 0:
            raise Exception("ERROR: must have at least one grade in the highest and lowest buckets to set the curve")
            
        num_scores = len(scores)
        scores.sort()
        num_lower_scores = 0
        self.cutoff_scores = []
        for grade in possible_grades[:-1]:
            num_lower_scores += grade_counts[grade]
            self.cutoff_scores.append(float(scores[num_lower_scores-1] + scores[num_lower_scores]) / 2)
            
    def set_curve_with_probs(self, scores, grade_probs):
        """Find the score cutoffs that separate each of the different possible grades."""
        possible_grades = grade_probs.keys()
        possible_grades.sort()
        last_grade = possible_grades[0]
        for grade in possible_grades[1:]:
            for i in range(last_grade + 1, grade):
                grade_probs[i] = 0
                #print "WARNING: did not specify prob for grade %d; only saw these grades: %s" %(last_grade + 1, str(possible_grades))
            last_grade = grade
        self.min_grade = possible_grades[0]
        self.max_grade = possible_grades[-1]
        
        if grade_probs[possible_grades[0]] * grade_probs[possible_grades[-1]] == 0:
            raise Exception("ERROR: must non-zero probability in the highest and lowest buckets to set the curve")
            
        num_scores = float(len(scores))
        scores.sort()
        num_lower_scores = 0
        cur_grade = self.min_grade
        cum_prob = grade_probs[self.min_grade]
        self.cutoff_scores = []
        last_score = scores[0]
        for score in scores:
            num_lower_scores += 1
            cum_frac = num_lower_scores / num_scores
            
            if cum_frac > cum_prob:                
                # # Find the cutoff score that splits the probability correctly
                # a = grade_probs[cur_grade] - (num_lower_scores-1)/num_scores
                # b = num_lower_scores/num_scores - grade_probs[cur_grade]
                # self.cutoff_scores.append((a*score + b*last_score) / (a + b))
                
                # Choose a cutoff in the middle of the two points
                self.cutoff_scores.append(float(score + last_score) / 2)
                
                
                cur_grade += 1
                if cur_grade == self.max_grade:
                    return
                while grade_probs[cur_grade] == 0:
                    self.cutoff_scores.append(self.cutoff_scores[-1])
                    cur_grade += 1
                cum_prob += grade_probs[cur_grade]
            
            last_score = score
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
