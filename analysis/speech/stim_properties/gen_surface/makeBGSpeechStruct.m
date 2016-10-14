function sstx = makeBGSpeechStruct(dirName)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Makes structure as .mat file using only /b/ and /g/ pairs for SpeechSearch that contains
%   -Basic stim information (speaker/consonant/vowel/etc.)
%   -Spectrogram for each phoneme
%   -Graph with similarity weights to other phonemes
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get filenames
fileList = getPhoPhiles(dirName);
cvList = {};
for i = 1:length(fileList)
    pathParts = strsplit(fileList{i},'/');
    if ~isempty(strmatch(pathParts(end-2),{'CV'},'exact')) && isempty(strmatch(pathParts(end-3),{'Ellen'},'exact')) && ...
            sum(strcmp(pathParts{end}(1),{'b','g'}))
        cvList = [cvList,fileList{i}];
    end
end
fileList = cvList; %Dumb way to do this but it seems to work
numFiles = length(fileList);
sstx = struct;

%Build structure
%Folder/file structure of input dir should be:
%   <dir>/Speaker/Phoneme Class (eg. CV for consonant
%   Vowel)/Phoneme/Phoneme#.wav
%   eg: '/Jonny/CV/bI/bI3.wav'
%ProcPhoPhiles should be able to build that for you if you're starting with
%raw audio
prevCharCnt = 0;
fprintf('\n\n')
M = zeros(.5*96000,numFiles);
F = [];
Ch= [];
for i = 1:numFiles
    fprintf([repmat('\b',1,(21+length(num2str(i-1))+length(num2str(numFiles)))),'Processing file %d of %d '],i,numFiles)
    sstx(i).file = fileList{i};
    pathParts = strsplit(fileList{i},'/');
    sstx(i).speaker = pathParts(end-3); %Speaker should always be 3 steps up from the file
    sstx(i).phonClass = pathParts(end-2); %CV, CVC, etc.
    sstx(i).phoneme = pathParts(end-1);
    sstx(i).recnum = fileList{i}(end-4); %Since extensions should be 4 chars, number of recording should be here
    
    %Feature extraction fo clustering
    [a,fs] = audioread(fileList{i});
    %Size Chopping so all are 500ms
    lt = size(a,1);
    if lt < (fs*.5)
        a(lt:(fs*.5),1) = 0;
    elseif lt > (fs*.5)
        a = a(1:fs*.5);
    end
    sstx(i).features = stFeatureExtraction(a,fs,.010,.010); %Extract features after making length same.
    %Save matrices for spectral clustering
    M(:,i) = a; %Full audio file for absolute 
    E(:,i,1) = sstx(i).features(2,:); %Energy
    F(:,i,2) = sstx(i).features(3,:); %Energy Entropy
    F(:,i,3) = sstx(i).features(4,:); %Spectral Centroid
    F(:,i,4) = sstx(i).features(6,:); %Spectral Entropy
end
fprintf(' \n')
fprintf('Simple processing completed, beginning spectral clustering\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Make Similarity Matrices
%Inspired by http://www.mathworks.com/matlabcentral/fileexchange/34412-fast-and-efficient-spectral-clustering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Normalizing
%Normalize M
minData = min(M, [], 2);
maxData = max(M, [], 2);

r = (0-1) ./ (minData - maxData);
s = 0 - r .* minData;
M = repmat(r, 1, size(M, 2)) .* M + repmat(s, 1, size(M, 2));

%Normalize F
for i = 1:4
    minData = min(F(:,:,i),[],2);
    maxData = max(F(:,:,i),[],2);
    r = (0-1) ./ (minData - maxData);
    s = 0 - r .* minData;
    F(:,:,i) = repmat(r, 1, size(F, 2)) .* F(:,:,i) + repmat(s, 1, size(F, 2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute distance matrices
WA = squareform(pdist(M'));
WF1 = squareform(pdist(F(:,:,1)'));
WF2 = squareform(pdist(F(:,:,2)'));
WF3 = squareform(pdist(F(:,:,3)'));
WF4 = squareform(pdist(F(:,:,4)'));

% Apply Gaussian similarity function and normalize
sigma = 1;
WA = matNorm(simGaussian(WA, sigma));
WF1 = matNorm(simGaussian(WF1,sigma));
WF2 = matNorm(simGaussian(WF2,sigma));
WF3 = matNorm(simGaussian(WF3,sigma));
WF4 = matNorm(simGaussian(WF4,sigma));


%imagesc(WF4)

%Write
for i = 1:numFiles
    sstx(i).similarAbs = WA(:,i);
    sstx(i).similarNRG = WF1(:,i);
    sstx(i).similarNRGEnt = WF2(:,i);
    sstx(i).similarSpecCent = WF3(:,i);
    sstx(i).similarSpecEnt = WF4(:,i);
end

cd(dirName)
cd ..
mkdir('SimMats');
cd('SimMats');
matPath = [pwd,'/BGphoMat.mat'];
save(matPath,'sstx');

fprintf('Spectral clustering completed, saving struct as %s \n',matPath)

%Also save csvs of matrices
csvwrite([pwd,'/BGWAbs.csv'],WA);
csvwrite([pwd,'/BGWNRG.csv'],WF1);
csvwrite([pwd,'/BGWNRGEnt.csv'],WF2);
csvwrite([pwd,'/BGWSpec.csv'],WF3);
csvwrite([pwd,'/BGWSpecEnt.csv'],WF4);





