import glob
import re
for filename in glob.iglob('test/test_*.py'):
    filename = re.sub('[^/]+/', '', filename[:-3])
    # import pdb;pdb.set_trace()
    exec("import %s" %filename)
    exec("from %s import *" %filename)