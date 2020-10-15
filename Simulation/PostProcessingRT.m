function [ final_inlier , RANSAC_total,predict_num_obj ] = PostProcessingRT(points_3d_2d,K,num_obj,result,sigma)%(point_3d,K,f1match,f2match,I1,I2,result,sigma)

% set the threshold so that there are at least 3 bins with >20 predicted inliers

% initial threshold
threshold = 0.9;

% adjust the threshold so that there are at least 3 bins with >20 predicted inliers
while sum(sum((result>=threshold),2)>20)<3 && threshold>=0.6
    threshold = threshold - 0.05;
end

%threshold= 0.7;

result = (result>=threshold);

% Show the before-filtering result
[B,I] = sort(sum(result,2),'descend');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nor_bin_result = ( sum(result,2)/sum(sum(result)));
% figure;
% subplot(2,1,1)
% plot(1:20,nor_bin_result,'*','MarkerSize',10,'Linewidth',2);
% hold on
% 
% for i = 1:20
%     line([i i], [0 nor_bin_result(i)],'Linewidth',2)
% end
% 
% xlim([1,20])
% ylim([0,1])
% 
% hold on
% xlabel('Index of bin')
% ylabel('Normalized number of predicted inliers')
% set(gca,'FontSize',15)
% grid on
% 
% subplot(2,1,2)
% plot(1:20,sum(result,2),'*','MarkerSize',10,'Linewidth',2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% count the RANSAC iteration number
RANSAC_total = 0;


%% Implement the RANSAC starting from the bin with largest number of predicted inliers

% final result stored in this variable
filtered_result = zeros(size(result,1),size(result,2));

bin_result = sum(result,2);

[B,I] = sort(bin_result,'descend');


for i = 1:20
    

    % skip if the number of inlier is <10
    predicted_inlier = result(I(i),:);
    if sum(predicted_inlier)<10
        continue
    end
    
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
    
    % count the iteration number 
    RANSAC_total = RANSAC_total + sample_count;
    

    if isempty(H_m_c) || sum(indices)<6
        %fprintf('Couldn''t find a pose\n');
        continue
    end
    
    % update the result
    ori_index = find(predicted_inlier);
    
    temp = zeros(1,size(result,2));
    temp(ori_index(indices))=1;
    
    filtered_result(I(i),:) = temp;
    
    % allocate the space for new inliers
    new_inlier = zeros(1,size(predicted_inlier,2));
    
    % check the pose for predicted inliers in other bins with >10 number of inliers 
    for j = 1:20
        
        
        pOutlier = 1/(w*h);
        
        % skip if the number of predicted inlier is <10
        predicted_inlier = result(I(j),:)-filtered_result(I(j),:);
        %sum(predicted_inlier)
        if  i==j%sum(predicted_inlier)<10 ||
            continue
        end
        
        % extract the corresponding features in image1 and image2
        ps = points_3d_2d(predicted_inlier==1,:);
    
        % find the inliers via RANSAC
        N = size(ps,1);    % Number of corresponding point pairs
    
        % extract the 3d and 2d points
        p3d = [ps(:,1:3)';ones(1,N)];
        p2d = K*[ps(:,4:5)';ones(1,N)];
        
        sigmas = sigma*ones(2,N);    % Sigma of errors of points found in lowest scale
       
        
        p_c = H_m_c * p3d;      % 3D points in camera coordinates
        pimg = K * p_c(1:3,:);  % Corresponding image points
        pimg(1,:) = pimg(1,:)./pimg(3,:);
        pimg(2,:) = pimg(2,:)./pimg(3,:);
        
        dp = p2d(1:2,:) - pimg(1:2,:);  % Diff btwn predicted and measured
        
        % Compute the probability that each residual error, dpi, could be an
        % inlier.  This is the equation of a Gaussian; ie.
        %   G(dpi) = exp(-dpi' * Ci^-1 * dpi)/(2*pi*det(Ci)^0.5)
        % where Ci is the covariance matrix of residual dpi:
        %   Ci = [ sxi^2  0;  0  syi^2 ]
        % Note that det(Ci)^0.5 = sxi*syi
        dp = dp ./ sigmas;      % Scale by the sigmas
        rsq = dp(1,:).^2 + dp(2,:).^2;    % residual squared distance errors
        numerator = exp(-rsq/2);    % Numerator of Gaussians, size is 1xN
        denominator = 2*pi*sigmas(1,:).*sigmas(2,:);  % Denominator, size 1xN
        
        % These are the probabilities of each point, if they are inliers (ie,
        % this is just the Gaussian probability of the error).
        pInlier = numerator./denominator;
        
        % Let's define inliers to be those points whose inlier probability is
        % greater than the outlier probability.
        indicesInliers = (pInlier > pOutlier);
        
        % update the result in the bin
        ori_index = find(predicted_inlier);
        
        temp = zeros(1,size(result,2));
        temp(ori_index(indicesInliers)) = 1;
        
        result(I(j),:) = result(I(j),:) - temp;
        result(I(i),:) = result(I(i),:) + temp;
        
        new_inlier = new_inlier + temp;
        
    end
    
    % add data in other bins but with the same transform to the current bin
    filtered_result(I(i),:) = filtered_result(I(i),:) + new_inlier;
    
end


% Show the after filtering result

result = filtered_result;

% when the normalized number of predicted inlier (0-1) is larger than 0.2
predict_num_obj = sum( ( sum(result,2)/sum(sum(result)) )>0.1 );
%fprintf('Image-2 contains: %d object(s)\n',predict_num_obj)

nor_bin_result = ( sum(result,2)/sum(sum(result)));
[B,I] = sort(nor_bin_result,'descend');

% construct the prediction output
final_inlier = zeros(num_obj,200);

for i = 1:num_obj
   
    final_inlier(i,:) = result(I(i),:);
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% nor_bin_result = ( sum(result,2)/sum(sum(result)));
% [B,I] = sort(nor_bin_result,'descend');
% 
% % show the bin prediction result
% figure;plot(1:20,( sum(result,2)/sum(sum(result))),'*','MarkerSize',10,'Linewidth',2);
% hold on
% for i = 1:20
%     line([i i], [0 nor_bin_result(i)],'Linewidth',2)
% end
% xlim([1,20])
% ylim([0,1])
% 
% hold on
% plot(1:20,ones(1,20)*0.1,'--','Linewidth',2);
% xlabel('Index of bin')
% ylabel('Normalized number of predicted inliers')
% set(gca,'FontSize',15)
% grid on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end