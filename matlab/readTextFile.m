function out = readTextFile(fileName,makeMat)
% if makeMat=1, we give back a matrix
fid = fopen(fileName, 'r');
fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 99900000);
fclose(fid);
fileLines = fileLines{1};

if exist('makeMat','var') && makeMat ==1
%     out = [];
%     for li = 1:length(fileLines)
%         [~, ~, ~, ~, ~, ~, splitLine] = regexp(fileLines{li}, ' ');
%         out = [out ; cellfun(@str2num,splitLine(1:end-1))];
%     end
    [~, ~, ~, ~, ~, ~, splitLine] = regexp(fileLines{1}, '\t');
    out = readmtx(fileName,length(fileLines),length(splitLine)-1,'%g');
else
    out = fileLines ;
end

% % % %followed for instance by:
% for li = 1:length(fileLines)
%     [~, ~, ~, ~, ~, ~, splitLine] = regexp(fileLines{li}, '\t');
% %...
% end