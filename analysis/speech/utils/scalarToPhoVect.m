function [phoVects] = scalarToPhoVect(varargin)
%Need to be able to match toneFreq vector representation of phoneme
%recordings to the scalar list that phoMats and similarity matrices use. 
%
%Arguments:
%phoMat - a phoMat made with makeSpeechStruct
%names, map - the names and map used in calcStim to present phonemes as
%cell arrays of strings

switch nargin
    case 1
        phoMat = varargin{1};
        names = {'Jonny','Ira','Anna','Dani','Theresa'};
        map = {'gI', 'go', 'ga', 'gae', 'ge', 'gu'; 'bI', 'bo', 'ba', 'bae', 'be', 'bu'};
    case 3
        phoMat = varargin{1};
        names = varargin{2};
        map = varargin{3};
    otherwise
        error('Need either just the phoMat or the phoMat, names, and map\n');    
end

%Fill in phoVects line by line
phoVects = zeros(length(phoMat),4);
for i = 1:length(phoMat)
    phoChar = char(phoMat(i).phoneme);
    [cons,vow] = find(strcmp(phoChar,map));
    phoVects(i,1) = cons;
    phoVects(i,3) = vow;
    
    phoName = char(phoMat(i).speaker);
    phoVects(i,2) = find(strcmp(phoName,names));
    
    phoVects(i,4) = str2num(phoMat(i).recnum);
end

%Save phoVects
%dataDir hardcoded for now... 