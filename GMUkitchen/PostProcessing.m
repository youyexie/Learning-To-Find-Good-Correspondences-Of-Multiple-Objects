function PostProcessing(f1match,f2match,I1,I2,result)

% set the threshold so that there are at least 3 bins with >20 predicted inliers

% initial threshold
threshold = 0.9;

% adjust the threshold so that there are at least 3 bins with >20 predicted inliers
while sum(sum((result>=threshold),2)>20)<3 && threshold>=0.7
    threshold = threshold - 0.05;
end

result = (result>=threshold);

% Show the before-filtering result
[B,I] = sort(sum(result,2),'descend');

nor_bin_result = ( sum(result,2)/sum(sum(result)));
figure;plot(1:20,nor_bin_result,'*','MarkerSize',10,'Linewidth',2);
hold on

for i = 1:20
    line([i i], [0 nor_bin_result(i)],'Linewidth',2)
end

xlim([1,20])
ylim([0,1])

hold on
xlabel('Index of bin')
ylabel('Normalized number of predicted inliers')
set(gca,'FontSize',15)
grid on

% % show the predicted inlier on the image
% figure; imshow([I1;I2],[]);
%
% for i = 1:num_obj
%
%     predicted_inlier = result(I(i),:);
%
%     % extract the corresponding features in image1 and image2
%     p1 = f1match(:,predicted_inlier==1);
%     p2 = f2match(:,predicted_inlier==1);
%
%     o = size(I1,1) ;
%
%     switch i
%         case 1
%             line([p1(1,:);p2(1,:)], ...
%             [p1(2,:);p2(2,:)+o],'Color','r') ;
%         case 2
%             line([p1(1,:);p2(1,:)], ...
%             [p1(2,:);p2(2,:)+o],'Color','b') ;
%         case 3
%             line([p1(1,:);p2(1,:)], ...
%             [p1(2,:);p2(2,:)+o],'Color','g') ;
%         otherwise
%             line([p1(1,:);p2(1,:)], ...
%             [p1(2,:);p2(2,:)+o],'Color','y') ;
%     end
%
% end



%% Implement the RANSAC starting from the bin with largest number of predicted inliers

% final result stored in this variable
filtered_result = zeros(size(result,1),size(result,2));

[w,h,d] = size(I1);

bin_result = sum(result,2);

[B,I] = sort(bin_result,'descend');


for i = 1:20
    
    % skip if the number of inlier is <10
    predicted_inlier = result(I(i),:);
    if sum(predicted_inlier)<10
        continue
    end
    
    % extract the corresponding features in image1 and image2
    p1 = f1match(:,predicted_inlier==1);
    p2 = f2match(:,predicted_inlier==1);
    
    % find the inliers via RANSAC
    N = size(p1,2);    % Number of corresponding point pairs
    
    sigma = 1;    % Sigma of errors of points found in lowest scale
    
    % Let's say that we want to get a good sample with this probability.
    confidence = 0.99;
    
    % Try to fit a homography from the points in image 1, to the points in
    % image 2. Returns:
    %   indices:  the indices of the original points, that are inliers
    %   rmsErr:  root mean squared error of the inliers (in pixels)
    %   tform_1_2:  a "projective2d" object, computed by "fitgeotrans".
    % Note: if you want to get the 3x3 homography matrix such that p2 = H*p1,
    % do Hom_1_2 = tform_1_2.T'.
    [tform_1_2, indices, rmsErr] = fitHomographyRansac( ...
        p1, ...    % image 1 keypoints, size 4xN
        p2, ...    % image 2 keypoints, size 4xN
        size(I2,1),size(I2,2), ...    % height and width of image 2
        sigma, ...      % uncertainty of image points at lowest scale
        1000, ...       % don't go above this many iterations
        confidence,  ... % desired confidence level, 0..1
        I1, I2 ...      % for visualization (use [],[] if not needed)
        );
    
    if isempty(tform_1_2)
        fprintf('Couldn''t find a homography\n');
        result(I(i),:) = 0;
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
        
        % skip if the number of predicted inlier is <10
        predicted_inlier = result(I(j),:)-filtered_result(I(j),:);
        %sum(predicted_inlier)
        if sum(predicted_inlier)<10 || i==j
            continue
        end
        
        % extract the corresponding features in image1 and image2
        p1 = f1match(:,predicted_inlier==1);
        p2 = f2match(:,predicted_inlier==1);
        
        
        pts1 = p1(1:2,:);  % Get x,y coordinates from image 1, size is (2,N)
        pts2 = p2(1:2,:);  % Get x,y coordinates from image 2, size is (2,N)
        
        % Now, if a point is an outlier, its residual error can be anything; i.e.,
        % any value between 0 and the size of the image.  Let's
        % assume a uniform probability distribution.
        pOutlier = 1/max(w,h);
        
        % Estimate the uncertainty of point locations.
        % Points found in higher scales will have proportionally larger errors.
        % Create a 2xN matrix of sigmas for each point, where each column is the
        % sigmax, sigmay values for that point.  Assume uncertainty is the same for
        % x and y.
        sigs = [sigma * p2(3,:); sigma * p2(3,:)];
        
        % Use that homography to transform all pts1 to pts2
        pts2map = transformPointsForward(tform_1_2, pts1');
        
        % Look at the residual errors of all the points.
        dp = (pts2map' - pts2);      % Size is 2xN
        
        % Compute the probability that each residual error, dp_i, could be an
        % inlier.  The probability is given by the equation of a Gaussian; ie.
        %   G(dp_i) = exp(-dp_i' * Ci^-1 * dp_i)/(2*pi*det(Ci)^0.5)
        % where Ci is the covariance matrix of residual dp_i:
        %   Ci = [ sx_i^2  0;  0  sy_i^2 ]
        % Note that det(Ci)^0.5 = sx_i*sy_i
        dp = dp ./ sigs;      % Scale by the sigmas
        rsq = dp(1,:).^2 + dp(2,:).^2;    % residual squared distance errors
        numerator = exp(-rsq/2);    % Numerator of Gaussians, size is 1xN
        denominator = 2*pi*sigs(1,:).*sigs(2,:);    % Denominator, size 1xN
        
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
num_obj = sum( ( sum(result,2)/sum(sum(result)) )>0.1 );
fprintf('Image-2 contains: %d object(s)\n',num_obj)

nor_bin_result = ( sum(result,2)/sum(sum(result)));
[B,I] = sort(nor_bin_result,'descend');

% show the bin prediction result
figure;plot(1:20,( sum(result,2)/sum(sum(result))),'*','MarkerSize',10,'Linewidth',2);
hold on
for i = 1:20
    line([i i], [0 nor_bin_result(i)],'Linewidth',2)
end
xlim([1,20])
ylim([0,1])

hold on
plot(1:20,ones(1,20)*0.1,'--','Linewidth',2);
xlabel('Index of bin')
ylabel('Normalized number of predicted inliers')
set(gca,'FontSize',15)
grid on

hold on

for i = 1:num_obj
    
    switch i
        case 1
            scatter(I(i),nor_bin_result(I(i)),1500,'x','LineWidth',10,'MarkerEdgeColor','r');
        case 2
            scatter(I(i),nor_bin_result(I(i)),1500,'x','LineWidth',10,'MarkerEdgeColor','b');
        case 3
            scatter(I(i),nor_bin_result(I(i)),1500,'x','LineWidth',10,'MarkerEdgeColor','g');
        otherwise
            scatter(I(i),nor_bin_result(I(i)),1500,'x','LineWidth',10,'MarkerEdgeColor','y');
    end
    
    
end
hold off

%% show the predicted inlier on the image
figure; imshow([I1;I2],[]);

for i = 4:num_obj
    
    predicted_inlier = result(I(i),:);
    
    % extract the corresponding features in image1 and image2
    p1 = f1match(:,predicted_inlier==1);
    p2 = f2match(:,predicted_inlier==1);
    
    o = size(I1,1) ;
    
    switch i
        case 1
            line([p1(1,:);p2(1,:)], ...
                [p1(2,:);p2(2,:)+o],'Color','r') ;
        case 2
            line([p1(1,:);p2(1,:)], ...
                [p1(2,:);p2(2,:)+o],'Color','b') ;
        case 3
            line([p1(1,:);p2(1,:)], ...
                [p1(2,:);p2(2,:)+o],'Color','g') ;
        otherwise
            line([p1(1,:);p2(1,:)], ...
                [p1(2,:);p2(2,:)+o],'Color','y') ;
    end
    
end

end