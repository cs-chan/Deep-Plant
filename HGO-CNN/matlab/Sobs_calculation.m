% 
% %%%%%%%%%%%% BD %%%%%%%%%%%%%
 clear all
 clc
 
 

dataset_folder = '/media/titanz/data3TB_02/PlantCLEF2015Test/img_256/testprob_256/';
 
 fprintf('Reading prob mat..\n');
 prob = dir(strcat(dataset_folder,'*.mat'));
 load('ClassID_CNNID.mat');
 
 fprintf('Loading author_list..\n');
 load('author_list.mat');
 
 S_individual = 0;
 
 for i = 1: length(author_list)
    S_overall = 0;
    num_obs = length(author_list(i).Obs_List);
    clear S_image_level
  
    S_individual = 0;
    for j = 1: num_obs      
       
       num_media = length(author_list(i).mediaC_List(j).media);
       J_obs_ave  = 0;
     
        
           for k = 1: num_media
               
            imagID = author_list(i).mediaC_List(j).media(k);
            groud_truth = author_list(i).mediaC_List(j).class(k);
            
            % read prob of that image
           probname  = strcat(sprintf('%04d',groud_truth),'-',num2str(ClassID_CNNID(groud_truth+1,1)),'-',num2str(imagID),'.jpg.mat');
            temp = load(strcat(dataset_folder,probname));
            fprintf('load %s\n',probname);
           J = temp.feature_map.prob;

            
 
            J_obs_ave = J_obs_ave + J;
           end
           J_obs_ave = J_obs_ave / num_media;
 
           
            [sorted_prob,predicted_classID] = sort(J_obs_ave,2,'descend'); 
 
            rank = find(predicted_classID == groud_truth + 1);
            
            S_individual  = S_individual + (1/rank);
           
   end
     
    S_image_level_perauthor(i) = sum(S_individual)/ num_obs;
 
     
 end
 S_obs = sum(S_image_level_perauthor) / length(author_list);


%%%%%%%%%%%% MAV %%%%%%%
clear all
clc


dataset_folder = '/media/titanz/data3TB_02/PlantCLEF2015Test/img_256/testprob_256/';

fprintf('Reading prob mat..\n');
prob = dir(strcat(dataset_folder,'*.mat'));
load('ClassID_CNNID.mat');

fprintf('Loading author_list..\n');
load('author_list.mat');

S_individual = 0;

for i = 1: length(author_list)
   S_overall = 0;
   num_obs = length(author_list(i).Obs_List);
   clear S_image_level
 
   S_individual = 0;
   for j = 1: num_obs      
      
      num_media = length(author_list(i).mediaC_List(j).media);

    
       J_obs_ave  = zeros(num_media,1000);
          for k = 1: num_media
              
           imagID = author_list(i).mediaC_List(j).media(k);
           groud_truth = author_list(i).mediaC_List(j).class(k);
           
           % read prob of that image
          probname  = strcat(sprintf('%04d',groud_truth),'-',num2str(ClassID_CNNID(groud_truth+1,1)),'-',num2str(imagID),'.jpg.mat');   
           temp = load(strcat(dataset_folder,probname));
           fprintf('load %s\n',probname);
          J = temp.feature_map.prob;

           

           J_obs_ave(k,:) = J;
          end
          

          if num_media > 1
          J_max_vot = max(J_obs_ave);
          else
              J_max_vot = J_obs_ave ;
          end
          
           [sorted_prob,predicted_classID] = sort(J_max_vot,2,'descend'); 

           rank = find(predicted_classID == groud_truth + 1);
           
           S_individual  = S_individual + (1/rank);
          
   end
    
   S_image_level_perauthor(i) = sum(S_individual)/ num_obs;

    
end
S_obs = sum(S_image_level_perauthor) / length(author_list);


