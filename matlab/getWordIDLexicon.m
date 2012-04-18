function wordIDLexicon = getWordIDLexicon(wordLexiconCounts,freqCut)

%go through sentences again, map words to their number and
% infrequent words (freq<=freqCut)to UNK
%list all words:
wordList = {};
wordCount = [];
for k = wordLexiconCounts.keys
    wordList{end+1} = k{1};
    wordCount(end+1) = wordLexiconCounts(k{1});
end
disp(['Total number of words:' num2str(length(wordCount))])
[vals ind] = sort(wordCount,'descend');

keepFreqWords = vals>=freqCut;
ind = ind(keepFreqWords);
wordList = wordList(ind);
wordCount = wordCount(ind);
disp(['Total number of words that appear at least ' num2str(freqCut) ' times:' num2str(length(wordCount))])
disp('Adding UNK')
wordList = ['UNK' wordList];
wordCount = [sum(~keepFreqWords) wordCount];

wordIDLexicon = containers.Map(wordList,1:length(wordList));