function wordLexicon = getAllWordCounts(fileLines,lowerCaseWords)

wordLexicon = containers.Map;

for li = 1:length(fileLines)
    if lowerCaseWords 
        [~, ~, ~, ~, ~, ~, sent] = regexp(lower(fileLines{li}), ' ');
    else
        [~, ~, ~, ~, ~, ~, sent] = regexp(fileLines{li}, ' ');
    end
%     % strip of sentence final punctuation
%     if strcmp(splitLine{2}(end),'.')
%         splitLine{2} = splitLine{2}(1:end-1);
%     end
    
    % very cheap tokenization: just ignore word final commas //and semi-colons
%     [~, ~, ~, ~, ~, ~, sent] = regexp(splitLine, ' ');
    % strip commas
    for w = sent
        if length(w{1})>1 && (strcmp(w{1}(end),',') || strcmp(w{1}(end),'.') || strcmp(w{1}(end),'?') || strcmp(w{1}(end),'!'))
            w{1} = w{1}(1:end-1);
        end
        if wordLexicon.isKey(w{1})
            wordLexicon(w{1})=wordLexicon(w{1})+1;
        else
            wordLexicon(w{1})=1;
        end
    end
end