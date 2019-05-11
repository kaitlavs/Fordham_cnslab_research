% Authors: Kaitlyn Lavan and Katie Mays
% last updated:  3/25/2019 by Kaitlyn Lavan - attempting to add a classifier

% VARIABLES
subjs = dir('/home/cnslab/cognition/Volumes/4*');                           %contains a struct of 117 subjects found in dir Volumes
Tasks = {'Ant''DgtSym' 'LetComp' 'LetSet' 'LogMem' 'MatReas'
         'PairAssoc' 'PaperFold' 'PattComp' 'Syn' 'WordOrder'};             %removed BehavioralFiles
all_ages = importdata('/home/cnslab/cognition/Volumes/RANN_AGES.csv');      %contains ages for all subjects, including those not found in dir Volumes
fail_subjs=zeros(length(subjs),length(Tasks));                              % matrix of zeros, with subjs = rows, Tasks = colms
index = 0;


 for i=1:length(subjs),
    subjNum(i)=str2num(subjs(i).name);                                      
    
    for j=1:length(Tasks),
        found=true;
        try
            dataMat=load_nii(['/home/cnslab/cognition/Volumes/' subjs(i).name '/' Tasks{j} '/FirstLevel/spm8/beta_0001.img' ]);
            brainData(i,j,:,:,:)=dataMat.img;
            
        catch
            % subjects who's beta_0001.img files did not load
            fail_subjs(i,j)=1;
            found=false;
        end;
    end;
    if found==true
        index=index+1;
        SubjectList{index}=subjs(i);
        ages(i)=all_ages.data(find(all_ages.data(:,1)==subjNum(i)),2);
    end;
 end;
 
 
% CLASSIFYING BRAINDATA

% subjects 1-50, task 5 -- training data
for i=1:50
   reshapeBrainData = reshape(brainData(i, 5, 60+[-2:2],50+[-2:2],10+[-2:2]), [1,125]);
   trainData(i,:) = reshapeBrainData;
   trainLabel(i)=5;
end;
 
% subjects 1-50, task 10 -- training data
for i=1:50
   var = brainData(i, 10, 60+[-2:2],50+[-2:2],10+[-2:2]);  %for some reason, had to save to its own variable to stop an indexing error
   reshapeBrainData = reshape(var, [1,125]);
   trainData(50+i,:) = reshapeBrainData;
   trainLabel(50+i)=10;
end;

taskClassifier = svmtrain(trainData, trainLabel);

% subjects 51-100, task 5 -- test data
for i=1:50 
    testData(i,:) = brainData(50+i, 5, 60+[-2:2],50+[-2:2],10+[-2:2]);
    testLabel(50+i)=5;
end;

possibleLabel = svmclassify(taskClassifier, testData);






% My code from 3/5
% for i=1:length(subjs),
%      for j=1:length(Tasks),
%        try
%           dataMat=load_nii(['/home/cnslab/cognition/Volumes/' subjs(i).name '/' Tasks{j} '/FirstLevel/spm8/beta_0001.img']);
%           fullData(i,j,:,:,:)=dataMat.img;
%           age_ind=find(ismember(all_ages.data, str2num(subjs(i).name)));
%           ages(i)= all_ages.data(age_ind, 2);
%           
%        end;
%     end;
% end;
