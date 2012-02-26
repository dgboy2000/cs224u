import hunspell

spellchecker = hunspell.hunspell()

input = "wat iz yo problem foo"
spellchecker.extractSpellingSuggestions(input)
numMistakes = spellchecker.getNumMistakes()
wordCount = spellchecker.getWordCount()
print input
print ("Spelling accuracy: " + str(wordCount-numMistakes) + "/" + str(wordCount))
