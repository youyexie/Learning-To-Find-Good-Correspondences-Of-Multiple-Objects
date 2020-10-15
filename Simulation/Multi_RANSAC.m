clear all; close all; clc

%% load the inference result

num_obj = 3;
std = 1;

inlier_portion = 0.3;

data = readNPY(strcat('TestData\Data_',num2str(num_obj),'object',num2str(inlier_portion),'std',num2str(std),'.npy'));
gt_label = readNPY(strcat('TestData\Label_',num2str(num_obj),'object',num2str(inlier_portion),'std',num2str(std),'.npy'));

inference = ones([1000,200]);
%% the inference result for object 1
%sigma = 1.0;

[batch_size , no_use ] = size(gt_label );

accuracy = zeros(batch_size,1);
detected_por = zeros(batch_size,1);
RANSAC_iter = zeros(batch_size,1);

K = [800 0 320;0 800 240;0 0 1];

tic
for i = 1:batch_size
    
    fprintf('Dealing with the %d-th sample\n',i);
    
    points_3d_2d = reshape(data(i,:,:,:),[200,5]);
    result = inference(i,:);
    
    % always detect 3 object 
    [predict_inlier , RANSAC_total] = PostProcessing(points_3d_2d,K,num_obj,result,1);
    
    gt_inlier = reshape(gt_label(i,:,:),[num_obj,200]);
    
    % change the order so that the objects' order matches the ground truth
    % label
    order_index = [];
    for j = 1:num_obj
        [B,I] = max( sum((gt_inlier(j,:).*predict_inlier)') );
        order_index = [order_index , I];
    end
    predict_inlier = predict_inlier(order_index,:);
    
    
    % calculate the accuracy and portion of detected inliers
    accuracy(i) = mean(sum(predict_inlier.*gt_inlier,2)./(sum(predict_inlier,2)+1e-8));
    detected_por(i) = mean(sum(predict_inlier.*gt_inlier,2)./sum(gt_inlier,2));
    RANSAC_iter(i) = RANSAC_total ;
end
toc

[mean(accuracy),mean(detected_por),mean(RANSAC_iter)]