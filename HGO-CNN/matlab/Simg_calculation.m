clear all
clc



dataset_folder = '/media/titanz/data3TB_02/PlantCLEF2015Test/img_256/testprob_256/';
fprintf('Reading prob mat..\n');
prob = dir(strcat(dataset_folder,'*.mat'));
load('ClassID_CNNID.mat');

fprintf('Loading author_list..\n');
load('author_list.mat');

S_individual = 0;
total_test_img = 0;
Acc =0;
maxlist = 0;
for i = 1: length(author_list)
   S_overall = 0;
   num_obs = length(author_list(i).Obs_List);
   clear S_image_level
   total_num_media = 0;
   
   for j = 1: num_obs      
      S_individual = 0;
      num_media = length(author_list(i).mediaC_List(j).media);
      if num_media > maxlist
          maxlist = num_media;
      end

       total_num_media = total_num_media + num_media;
          for k = 1: num_media
           total_test_img = total_test_img + 1 ;   
           imagID = author_list(i).mediaC_List(j).media(k);
           groud_truth = author_list(i).mediaC_List(j).class(k);
           
           % read prob of that image
           probname  = strcat(sprintf('%04d',groud_truth),'-',num2str(ClassID_CNNID(groud_truth+1,1)),'-',num2str(imagID),'.jpg.mat');
           temp = load(strcat(dataset_folder,probname));
           fprintf('load %s\n',probname);
           J = temp.feature_map.prob;


           
           [sorted_prob,predicted_classID] = sort(J,2,'descend'); % B store the prob output while I store the classidx
           if predicted_classID(1,1) == groud_truth + 1
           Acc = Acc +1;   
           end
           rank = find(predicted_classID == groud_truth + 1);

           %S_overall  = S_overall + (1/rank);  
           
           S_individual  = S_individual + (1/rank);
          end 
      S_individual  = S_individual /  num_media;  
      S_image_level(j) =   S_individual;  
          
   end
   
   
    
   S_image_level_perauthor(i) = sum(S_image_level)/ num_obs;

    
end
S_image = sum(S_image_level_perauthor) / length(author_list);
Acc_final = Acc / total_test_img;



