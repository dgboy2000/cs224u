function [X_trn_text,X_tst_text] = extractTextFeatures(Text_trn,Text_tst,prefix2)


if ~exist([prefix2 '.textPrePro.train.mat'],'file')
    lowerCaseWords=0;
    freqCut=2;
    wordLexiconCounts = getAllWordCounts(Text_trn,lowerCaseWords);
    wordIDLexicon = getWordIDLexicon(wordLexiconCounts,freqCut);
    
    allKeys = wordIDLexicon.keys;
    for k=1:length(allKeys)
        words{wordIDLexicon(allKeys{k})} = allKeys{k};
    end
    
    [allSNum, allSStr, allSOStr] = preProLines(Text_trn,wordIDLexicon,lowerCaseWords);
    save([prefix2 '.textPrePro.train.mat'],'allSNum','allSStr', 'allSOStr','wordIDLexicon','words');
    [allSNumTest, allSStrTest, allSOStrTest] = preProLines(Text_tst,wordIDLexicon,lowerCaseWords);
    save([prefix2 '.textPrePro.test.mat'],'allSNumTest','allSStrTest', 'allSOStrTest','words');
else
    load([prefix2 '.textPrePro.train.mat'],'allSNum','allSStr', 'allSOStr','wordIDLexicon');
    load([prefix2 '.textPrePro.test.mat'],'allSNumTest','allSStrTest', 'allSOStrTest');
end

X_trn_text = getFeat(allSNum,Text_trn,wordIDLexicon);
X_tst_text = getFeat(allSNumTest,Text_tst,wordIDLexicon);


function X_trn_text = getFeat(allSNum,Text_trn,wordIDLexicon)

% list of features to include:
includeFullHist = 0;
nMostFreqCounts = 20;
specialTokens = {'.' '?' '!' '-' ';'};
histHistCounts = 10;


numWords = double(wordIDLexicon.Count);
numFeat = nMostFreqCounts + length(specialTokens) + histHistCounts;
if includeFullHist
    numFeat = numFeat+ numWords;
end

numEss = length(allSNum);



X_trn_text = zeros(numEss,numFeat);

for i = 1:numEss
    H = hist(allSNum{i},1:numWords);
    % counts of n most frequent words (bad quality to repeat yourself a lot)
    [val ind]= sort(H,'descend');
    X_trn_text(i,1:nMostFreqCounts) = val(1:nMostFreqCounts);
    featInd = nMostFreqCounts+1;
    
    % entropy of normalized histogram
    normHist = H./sum(H);
    X_trn_text(i,featInd) = entropy(normHist);
    featInd=featInd+1;
    for t = specialTokens
        X_trn_text(i,featInd) = length(findstr(Text_trn{i},t{1}));
        featInd=featInd+1;
    end
    
    % histogram of words counts (how many times are words used n times)
    HH = hist(H(H>0),1:histHistCounts);
    X_trn_text(i,featInd:featInd+histHistCounts-1) = HH;
    featInd = featInd + histHistCounts-1;
    
    if includeFullHist
        % the normalized histogram
        X_trn_text(i,featInd:featInd+length(H)-1) = normHist;
        featInd=featInd+length(H);
    end
    
    assert(featInd==size(X_trn_text,2));
end