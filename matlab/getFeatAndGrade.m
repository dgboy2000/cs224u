function [X_trn Y_trn] = getFeatAndGrade(prefix,prefix2,tt)

X_trn = readTextFile([prefix '.' tt],1);
gradeAndEssay_trn = readTextFile([prefix2 '.' tt]);
Y_trn = zeros(length(gradeAndEssay_trn)-1,1);
for li = 2:length(gradeAndEssay_trn)
    [~, ~, ~, ~, ~, ~, splitLine] = regexp(gradeAndEssay_trn{li}, '\t');
    Y_trn(li-1) = str2double(splitLine{1});
end

% TODO: Essay analysis in gradeAndEssay