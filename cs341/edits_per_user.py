# From http://mail.python.org/pipermail/python-dev/2000-October/009946.html

import re
import sys, string
from xml.sax import saxutils, handler, make_parser

class RevisionHandler(handler.ContentHandler):
  def __init__(self):
      handler.ContentHandler.__init__(self)
      self._curPage = None
      self._idType = None
      self._inContributorTag = False
      self._revisionAuthors = []

  # ContentHandler methods
  def startElement(self, name, attrs):
    if name.lower() == 'contributor':
      self._inContributorTag = True
    elif self._inContributorTag and name.lower() in ('id', 'ip'):
      self._idType = name.lower()

  def endElement(self, name):
    if name.lower() == 'contributor':
      self._inContributorTag = False
    elif name.lower() == self._idType:
      self._idType = None

  def characters(self, content):
    if self._inContributorTag and self._idType is not None:
      self._revisionAuthors.append(content)
      if len(self._revisionAuthors) % 10000 == 0:
        print "Processed %d revisions" %len(self._revisionAuthors)

# --- The main program
def testXML():
  parser = make_parser()
  revisionHandler = RevisionHandler()
  parser.setContentHandler(revisionHandler)
  parser.parse("/Users/dannygoodman/Downloads/enwiki-20111201-stub-meta-history1.xml")
  print "Processed %d total revisions" %len(revisionHandler._revisionAuthors)
  
  #!/usr/bin/python
  import sys 
  import re 

def main(argv): 
  pattern = re.compile("^        <i(?:d|p)>(\d+)</i(?:d|p)>\s*$") 
  for line in sys.stdin: 
    match = pattern.match(line)
    if match is not None:
      print "LongValueSum:" + match.group(1) + "\t" + "1" 




if __name__ == "__main__": 
    main(sys.argv)








