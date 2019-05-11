% Authors: Kaitlyn Lavan and Katie Mays
% last updated:
% 
% 4/15/2019 by Kaitlyn Lavan - added tic toc with unsubstantial time (unsolved problem of what is taking so long)
%        - changed  Maxiters default for svm train to see if timing changed
%	 - added 3D for loop (x,y,z) 
%	 - and accuracy matrix
%	 - swapped svmtrain with fitcsvm and svmclassify with predict !!! only to be used on 2018 and up
% 4/06/2019 by Katie Mays - fixed dimension errors in training for-loop  
% 4/01/2019 by Katie Mays - added back 'behavioralFiles' from Tasks  because it was offsetting everything by 1
%   after calling reshape braindata, failed subjects shrinks to 5 colums
%   from 11, this issue is fixed by making copy of brain data we need to
%   work with. i also just combined the learning into one for loop
%   currently an error in svmtrain apparently? "TRAINING and Y must both
%   contain at least one row with no missing values"
% 3/25/2019 by Kaitlyn Lavan - set up data to be classified, error with failed
%   subjects
% 3/15/2019 by Katie Mays - added try catch block 
% 3/07/2019 by Kaitlyn Lavan -  uploaded and imported subject data, created
%   failed subject matrix to contain all subjects that do not contain data
%   for a specified task                          

% REMINDER : run 'matlabSetup' in matlab /cnslab before running this 
%


% VARIABLES
subjs=dir('/home/cnslab/cognition/Volumes/4*');                      %contains a struct of 118 subjects found in dir Volumes
Tasks={'Ant''BehavioralFiles' 'DgtSym' 'LetComp' 'LetSet' 'LogMem' 'MatReas' 'PairAssoc' 'PaperFold' 'PattComp' 'Syn' 'WordOrder'};
all_ages=importdata('/home/cnslab/cognition/Volumes/RANN_AGES.csv'); %contains ages for all subjects, including those not found in dir Volumes

                                                                      % filling fail_subjs first greatly reduced the running time
fail_subjs=zeros(length(subjs),length(Tasks));                        % matrix of zeros, with subjs = rows, Tasks = colms
index=0;
max_ind=0;

 tic
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
 toc
 
% CLASSIFYING BRAINDATA

%     starting number:increments:ending number
for x=5:5:70,
   for y=5:5:80,
      for z=5:5:60,
	% subjects 1-50 training data
	fprintf('for loop training data with reshape(): '); %Elapsed time is 0.005361 seconds.
	tic
	for i=1:50
   	    % create copies of brainData necessary because otherwise reshape erases data 
   	    task5img=brainData(i,5,x+[-2:2],y+[-2:2],z+[-2:2]);
   	    task10img=brainData(i,10,x+[-2:2],y+[-2:2],z+[-2:2]);

   	    % training data for task 5
   	    reshapeBrainTask5=reshape(task5img, [1,125]);
   	    trainData(i,:)=reshapeBrainTask5;
   	    trainLabel(i)=5;
   	    % training data for task 10
   	    reshapeBrainTask10=reshape(task10img, [1,125]);
   	    trainData(50+i,:)=reshapeBrainTask10;
    	    trainLabel(50+i)=10;
   	   % ... add more training data?
	end;
	toc
    
	fprintf('svm train: '); %Elapsed time is 0.524263 seconds.
	tic
	taskClassifier=fitcsvm(trainData, trainLabel);
	toc


	fprintf('for loop testing data with reshape(): '); %Elapsed time is 0.004296 seconds.
	tic
	% subjects 51-100 testing data
	for i=1:50
	    task5img=brainData(i,5,x+[-2:2],y+[-2:2],z+[-2:2]);
    	    task10img=brainData(i,10,x+[-2:2],y+[-2:2],z+[-2:2]);

	    % testing data for task 5
	    reshapedTesting5=reshape(task5img, [1,125]);
	    testData(i,:)=reshapedTesting5;
	    testLabel(i)=5; 
	    % testing data for task 10
	    reshapedTesting10=reshape(task10img, [1,125]);
	    testData(50+i,:)=reshapedTesting10;
	    testLabel(50+i)=10; 
	    % ... add more training data?
	end;
	toc



	fprintf('svm classify: '); %Elapsed time is 0.043698 seconds.
	tic
	possibleLabel=(predict(taskClassifier, testData))';
	toc

	% compare possible label with train label -- create 4D accuracy matrix with (x, y, z)
	numCorrect=length(find(possibleLabel==trainLabel));
	accMat(x,y,z)=(numCorrect/length(trainLabel))*100; %as percent

	if (accMat(x,y,z) > max_ind)
	   max_ind=accMat(x,y,z);
	   x_ind=x;
	   y_ind=y;
	   z_ind=z;
	end;
      end;
   end;
end;
