import glob
for filename in glob.iglob('test_*.py'):
    filename = filename[:-3]
    exec("import %s" %filename)
    exec("from %s import *" %filename)