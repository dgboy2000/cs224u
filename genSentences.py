import DataSet
import nltk

def cleanUpSentence(line):
    line = line.replace('.', '. ')
    line = line.replace('?', '? ')
    line = line.replace('!', '! ')
    line = line.replace(':', ': ')
    line = line.replace(';', '; ')
    line = line.replace(',', ', ')
    line = line.replace('  ', ' ')
    line = line.replace('. .', '..')
    line = line.replace('? ?', '??')
    line = line.replace('! !', '!!')
    line = line.replace('. com', '.com')
    line = line.replace('. net', '.net')
    line = line.replace('. edu', '.edu')
    line = line.replace('. org', '.org')
    return line


for essay_set in range(1, 9):
    ds = DataSet.DataSet()
    ds.importData('data/c_train.utf8ignore.tsv', essay_set)

    f = open('socher/train/essay_set_%d.tsv' % essay_set, 'w')

    text = ds.getRawText()
    grade = ds.getGrades()
    for i in range(0, ds.size()):
        line = text[i]
        line = cleanUpSentence(line) 
        
        for sentence in nltk.PunktSentenceTokenizer().tokenize(line):
            if len(line) > 1:
                f.write("%d\t%s\n" %(grade[i], sentence))

for essay_set in range(1, 9):
    ds = DataSet.DataSet()
    ds.importData('data/c_val.utf8ignore.tsv', essay_set)

    f = open('socher/test/essay_set_%d.tsv' % essay_set, 'w')

    text = ds.getRawText()
    grade = ds.getGrades()
    for i in range(0, ds.size()):
        line = text[i]

        line = cleanUpSentence(line) 

        for sentence in nltk.PunktSentenceTokenizer().tokenize(line):
            if len(sentence) > 1:
                f.write("%d\t%s\n" %(grade[i], sentence))

