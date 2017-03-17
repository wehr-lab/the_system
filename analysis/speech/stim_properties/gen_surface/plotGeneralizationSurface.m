function plotGeneralizationSurface(phoMat,csvDir)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots a 2d MDS of the chosen similarity matrices from phoMat
% in the x-y plane, and a kernel density estimation of % correct in the z
% axis. Used to assess whether the similarity matrix is a good fit for the
% generalization data
%
%
% Arguments:
% phoMat - /b/ /g/ phoMat created by makeBGSpeechStruct
% phoMat - phoMat created by makeCVSpeechStruct
% csvDir - directory of compiled trial record csv's as made by
% cleanPermanentRecords
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%What sim matrix?
simtype = input('What similarity matrix would you like to use? \n1)Waveform \n2)Energy \n3)Energy Entropy \n4)Spectral Centroid \n5)Spectral Entropy \n\n   >');
switch simtype
    case 1
        for i = 1:length(phoMat)
            simmat(1:length(phoMat),i) = phoMat(i).similarAbs;
        end
    case 2
        for i = 1:length(phoMat)
            simmat(1:length(phoMat),i) = phoMat(i).similarNRG;
        end        
    case 3
        for i = 1:length(phoMat)
            simmat(1:length(phoMat),i) = phoMat(i).similarNRGEnt;
        end        
    case 4
        for i = 1:length(phoMat)
            simmat(1:length(phoMat),i) = phoMat(i).similarSpecCent;
        end        
    case 5
        for i = 1:length(phoMat)
            simmat(1:length(phoMat),i) = phoMat(i).similarSpecEnt;
        end        
end

%turn into dissimilarity matrix
simmat = 1-simmat;

%MDS
fprintf('Performing MDS with 3 replicates\n');
tic
[MDmat,stress,dispar] = mdscale(simmat,2,'Criterion','sstress','Replicates',3);
fprintf('MDS Complete in %.1f seconds\n',toc);

%Get vectorform stim ID
phoVects = scalarToPhoVect(phoMat);

%Scatterplot by /b/ /g/
figure
subplot(1,4,1)
colorvec = zeros(length(phoVects),3);
colorvec(find(phoVects(:,1)==1),1) = 1;
colorvec(find(phoVects(:,1)==2),3) = 1;
scatter(MDmat(:,1),MDmat(:,2),20,colorvec,'filled')

%Scatterplot by vowel
subplot(1,4,2)
colorvec2 = colorvec;
colorvec2(:,2) = phoVects(:,3)./8;
scatter(MDmat(:,1),MDmat(:,2),20,colorvec2,'filled')

%Scatterplot by speaker
subplot(1,4,3:4)
colormap(jet)
hold on
for i = 1:length(unique(phoVects(:,2)))
    p(i) = scatter(MDmat(find(phoVects(:,2)==i),1),MDmat(find(phoVects(:,2)==i),2),20,repmat(i*7,length(find(phoVects(:,2)==i)),1),'filled')
end
hold off
%scatter(MDmat(:,1),MDmat(:,2),20,phoVects(:,2),'filled')
legend([p(1),p(2),p(3),p(4),p(5)],{'Jonny','Ira','Anna','Dani','Theresa'},'Location','EastOutside')


%Scatterplot by vowel
subplot(1,4,2)
colorvec2 = colorvec;
colorvec2(:,2) = phoVects(:,3)./8;
scatter(MDmat(:,1),MDmat(:,2),20,colorvec2,'filled')

%Scatterplot by speaker
subplot(1,4,3:4)
colormap(jet)
hold on
for i = 1:length(unique(phoVects(:,2)))
    p(i) = scatter(MDmat(find(phoVects(:,2)==i),1),MDmat(find(phoVects(:,2)==i),2),20,repmat(i*7,length(find(phoVects(:,2)==i)),1),'filled')
end
hold off
%scatter(MDmat(:,1),MDmat(:,2),20,phoVects(:,2),'filled')
legend([p(1),p(2),p(3),p(4),p(5)],{'Jonny','Ira','Anna','Dani','Theresa'},'Location','EastOutside')

