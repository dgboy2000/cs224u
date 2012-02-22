import unittest
from score import KappaScore, MeanKappaScore

class Testquadratic_weighted_kappa(unittest.TestCase):


    def test_confusion_matrix(self):
        score = KappaScore([1,2],[1,2])
        conf_mat = score.confusion_matrix()
        self.assertEqual(conf_mat,[[1,0],[0,1]])
        
        score = KappaScore([1,2],[1,2],0,2)
        conf_mat = score.confusion_matrix()
        self.assertEqual(conf_mat,[[0,0,0],[0,1,0],[0,0,1]])
        
        score = KappaScore([1,1,2,2,4],[1,1,3,3,5])
        conf_mat = score.confusion_matrix()
        self.assertEqual(conf_mat,[[2,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],
                                   [0,0,0,0,1],[0,0,0,0,0]])
        
        score = KappaScore([1,2],[1,2],1,4)
        conf_mat = score.confusion_matrix()
        self.assertEqual(conf_mat,[[1,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])

    def test_quadratic_weighted_kappa(self):
        kappa = KappaScore([1,2,3],[1,2,3]).quadratic_weighted_kappa()
        self.assertAlmostEqual(kappa, 1.0)

        kappa = KappaScore([1,2,1],[1,2,2],1,2).quadratic_weighted_kappa()
        self.assertAlmostEqual(kappa, 0.4)

        kappa = KappaScore([1,2,3,1,2,2,3],[1,2,3,1,2,3,2]).quadratic_weighted_kappa()
        self.assertAlmostEqual(kappa, 0.75)


    def test_mean_quadratic_weighted_kappa(self):
        class MockKappa(KappaScore):
            def __init__(self, val):
                self.val = val
            def quadratic_weighted_kappa(self):
                return self.val
            
        mks = MeanKappaScore()
        mks.add(MockKappa(1))
        mks.add(MockKappa(1))
        kappa = mks.mean_quadratic_weighted_kappa()
        self.assertAlmostEqual(kappa, 0.999)

        mks = MeanKappaScore()
        mks.add(MockKappa(0.5))
        mks.add(MockKappa(0.8))
        kappa = mks.mean_quadratic_weighted_kappa([1,.5])
        self.assertAlmostEqual(kappa, 0.624536446425734)

        mks = MeanKappaScore()
        mks.add(MockKappa(-1))
        mks.add(MockKappa(1))
        kappa = mks.mean_quadratic_weighted_kappa()
        self.assertAlmostEqual(kappa, 0.0)

if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    