function [ final_inlier , RANSAC_total ] = PostProcessing(points_3d_2d,K,num_obj,result,sigma)

% set the threshold so that there are at least 3 bins with >20 predicted inliers

% initial threshold
threshold = 0.5;

result = (result>=threshold);

% final result stored in this variable
final_inlier = zeros(num_obj,size(result,2));
RANSAC_total = 0;

%% Implement the RANSAC 

for i = 1:num_obj    

    % search in the rest
    predicted_inlier = result - sum(final_inlier);
    
    % extract the corresponding features in image1 and image2
    ps = points_3d_2d(predicted_inlier==1,:);
    
    % find the inliers via RANSAC
    N = size(ps,1);    % Number of corresponding point pairs
    
    % extract the 3d and 2d points
    p3d = [ps(:,1:3)';ones(1,N)];
    p2d = K*[ps(:,4:5)';ones(1,N)];
    
    sigmas = sigma*ones(2,N);    % Sigma of errors of points found in lowest scale
    w = K(1,3)*2;h = K(2,3)*2;
    % Try to fit a homography from the points in image 1, to the points in
    % image 2. Returns:
    %   indices:  the indices of the original points, that are inliers 
    %   rmsErr:  root mean squared error of the inliers (in pixels)
    %   tform_1_2:  a "projective2d" object, computed by "fitgeotrans".
    % Note: if you want to get the 3x3 homography matrix such that p2 = H*p1,
    % do Hom_1_2 = tform_1_2.T'.
    [sample_count,fSuccess, H_m_c, indices, rmsErr] = ...
    sub_Pose_from_2D_3D_Ransac(p2d, p3d, K, w,h, sigmas);
    
    RANSAC_total = RANSAC_total + sample_count;

    if isempty(H_m_c) || sum(indices)<3
        %fprintf('Couldn''t find a pose\n');
        continue
    end
    
    % update the result
    ori_index = find(predicted_inlier);
    
    temp = zeros(1,size(result,2));
    temp(ori_index(indices))=1;
    
   

    % update the result to the final output
    final_inlier(i,:) = temp;
    
end




end