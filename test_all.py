#!/usr/bin/python

import glob
import re
import unittest

test_file_strings = glob.glob('test/test_*.py')
module_strings = [re.sub('/','.',str[:-3]) for str in test_file_strings]
suites = [unittest.defaultTestLoader.loadTestsFromName(str) for str in module_strings]
testSuite = unittest.TestSuite(suites)
text_runner = unittest.TextTestRunner().run(testSuite)

print "Done"