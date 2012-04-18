function [allSNum, allSStr, allSOStr] = preProLines(fileLines,wordIDLexicon,lowerCaseWords)
for li = 1:length(fileLines)
    if lowerCaseWords
        [~, ~, ~, ~, ~, ~, sent] = regexp(lower(fileLines{li}), ' ');
    else
        [~, ~, ~, ~, ~, ~, sent] = regexp(fileLines{li}, ' ');
    end
    
    
    %labels(li) = str2double(splitLine{1});
    
%     % strip of sentence final punctuation
%     if strcmp(splitLine{2}(end),'.')
%         splitLine{2} = splitLine{2}(1:end-1);
%     end
%     
    % very cheap tokenization: just ignore word final commas and semi-colons
%     [~, ~, ~, ~, ~, ~, sent] = regexp(splitLine, ' ');
    allSNum{li} = [];
    allSStr{li} = {};
    allSOStr{li} = {};
    % strip commas
    for w = sent
        if length(w{1})>1 && strcmp(w{1}(end),',')
            w{1} = w{1}(1:end-1);
        end
        if length(w{1})==0
            continue;
        end
        allSOStr{li} = [allSOStr{li} w{1}];
        if wordIDLexicon.isKey(w{1})
            allSNum{li} = [allSNum{li} wordIDLexicon(w{1})];
            allSStr{li} = [allSStr{li} w{1}];
        else
            % UNK's the first word
            allSNum{li} = [allSNum{li} 1];
            allSStr{li} = [allSStr{li} 'UNK'];
        end
        
    end
end