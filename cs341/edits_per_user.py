# From http://mail.python.org/pipermail/python-dev/2000-October/009946.html

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

# # --- The main program
# 
# parser = make_parser()
# parser.setContentHandler(ContentGenerator())
# parser.parse(sys.argv[1])









