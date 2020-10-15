function [tform_1_2, indices, rmsErr] = ...
    fitHomographyRansac( ...
    keypts1, ...    % image 1 keypoints, size 4xN
    keypts2, ...    % image 2 keypoints, size 4xN
    H,W, ...        % height and width of image 2
    sigma, ...      % uncertainty of image points at lowest scale
    maxIterations, ...  % don't do more than this many iterations
    Ps,  ... % desired confidence level (prob of success); value from 0..1
    I1, I2 ...          % show images (for visualization only)
    )
% Fit homography to corresponding points.  Uses the MLESAC (m-estimator
% sample consensus) algorithm to find inliers.  See the paper: "MLESAC: A
% new robust estimator with application to estimating image geometry" by
% Torr and Zisserman, at
% http://www.robots.ox.ac.uk/~vgg/publications/2000/Torr00/torr00.pdf
% Output parameters:
%   tform_1_2:  Matlab "transform" object (created by "fitgeotrans"
%   indices:    Indices of inliers
%   rmsErr:     Root mean square error
% If no homography can be fit, the output parameters are empty.

tform_1_2 = [];
indices = [];
rmsErr = [];

N = size(keypts1,2);    % Number of corresponding point pairs
if N<4
    fprintf('Can''t fit a homography using less than 4 point pairs\n');
    return;
end

pts1 = keypts1(1:2,:);  % Get x,y coordinates from image 1, size is (2,N)
pts2 = keypts2(1:2,:);  % Get x,y coordinates from image 2, size is (2,N)

% Estimate the uncertainty of point locations.  
% Points found in higher scales will have proportionally larger errors.
% Create a 2xN matrix of sigmas for each point, where each column is the
% sigmax, sigmay values for that point.  Assume uncertainty is the same for
% x and y.
sigs = [sigma * keypts2(3,:); sigma * keypts2(3,:)];

% Now, if a point is an outlier, its residual error can be anything; i.e.,
% any value between 0 and the size of the image.  Let's
% assume a uniform probability distribution.
pOutlier = 1/max(W,H);

% Estimate the fraction of inliers.  We'll actually estimate this later
% from the data, but start with a worst case scenario assumption.
inlierFraction = 0.1;


% Determine the required number of iterations
nInlier = round(N*inlierFraction); % number of inliers among the N points
% Calculate the number of ways to pick a set of 4 points, out of N points.
% This is "N-choose-4", which is N*(N-1)*(N-2)*(N-3)/factorial(4).
m = nInlier*(nInlier-1)*(nInlier-2)*(nInlier-3)/factorial(4); % #ways to choose 4 inliers
if m<1  m=1;    end
n = N*(N-1)*(N-2)*(N-3)/factorial(4);  % #ways to choose 4 points

p = m/n;        % probability that any sample of 4 points is all inliers
nIterations = log(1-Ps) / log(1 - p);
nIterations = ceil(nIterations);
fprintf('Initial estimated number of iterations needed: %d\n', nIterations);
%pause

sample_count = 0;   % The number of Ransac trials
pBest = -Inf;       % Best probability found so far (actually, the log)
while sample_count < nIterations
    if sample_count > maxIterations
        break;
    end
    
    % Grab 4 matching points at random
    v = randperm(N);
    p1 = pts1(:,v(1:4));
    p2 = pts2(:,v(1:4));
    
    % Make sure the 4 points are not co-linear.  
    % Note: 3 points are colinear if  (p3-p1)x(p2-p1) = 0.
    q = [p1; ones(1,4)];
    if norm(cross(q(:,3)-q(:,1), q(:,2)-q(:,1))) < 1e-6 || ...
        norm(cross(q(:,4)-q(:,2), q(:,3)-q(:,2))) < 1e-6
        %disp('Image 1 points are degenerate:'), disp(p1);    
        continue;
    end
    q = [p2; ones(1,4)];
    if norm(cross(q(:,3)-q(:,1), q(:,2)-q(:,1))) < 1e-6 || ...
        norm(cross(q(:,4)-q(:,2), q(:,3)-q(:,2))) < 1e-6
        %disp('Image 2 points are degenerate:'), disp(p2);    
        continue;
    end
    
    % Try fitting a homography, that transforms p1 to p2.  Matlab will
    % display a warning if the result is close to singular, so turn
    % warnings temporarily off.
    warning('off', 'all');
    try
        tform_1_2 = fitgeotrans(p1',p2','projective');
    catch
        continue;
    end
    warning('on', 'all');
    
    % Ok, we were able to fit a homography to this sample.
    sample_count = sample_count + 1;
    
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
    nInlier = sum(indicesInliers);

        
    % Net probability of the data (actually, the log of the probability).
    p = sum(log(pInlier(indicesInliers))) + (N-nInlier)*log(pOutlier);
    
    % Keep this solution if probability is better than the best so far.
    if p>pBest
        pBest = p;
        indices = indicesInliers;
        fprintf(' (Iteration %d): best so far with %d inliers, p=%f\n', ...
            sample_count, nInlier, p);
        
        % Show inliers
        if ~isempty(I1) && ~isempty(I2)
            %figure(100), imshow([I1,I2],[]);
            o = size(I1,2) ;
            for i=1:size(pts1,2)
                x1 = pts1(1,i);
                y1 = pts1(2,i);
                x2 = pts2(1,i);
                y2 = pts2(2,i);

%                 if indicesInliers(i)
%                     text(x1,y1,sprintf('%d',i), 'Color', 'g');
%                     text(x2+o,y2,sprintf('%d',i), 'Color', 'g');
%                 else
%                     text(x1,y1,sprintf('%d',i), 'Color', 'r');
%                     text(x2+o,y2,sprintf('%d',i), 'Color', 'r');
%                 end
            end
        end
        %pause
    end
    
    % Update the number of iterations required if we got a lot of inliers.
    if nInlier/N > inlierFraction
        inlierFraction = nInlier/N;
        
        m = nInlier*(nInlier-1)*(nInlier-2)*(nInlier-3)/factorial(4);
        n = N*(N-1)*(N-2)*(N-3)/factorial(4);  % #ways to choose 4 points
        p = m/n;  % probability that any sample of 4 points is all inliers
        nIterations = log(1-Ps) / log(1 - p);
        nIterations = ceil(nIterations);
        fprintf(' (Iteration %d): New estimated number of iterations needed: %d\n', ...
            sample_count, nIterations);
    end

end
    
fprintf('Final number of iterations used: %d\n', sample_count);
if sum(indices) < 4
    return      % Couldn't find a fit
end
fprintf('Final calculated inlier fraction: %f\n', inlierFraction);

% Ok, refit homography using all the inliers.
p1 = pts1(:,indices);
p2 = pts2(:,indices);
tform_1_2 = fitgeotrans(p1',p2','projective');

% Determine the final residual error.
p2map = transformPointsForward(tform_1_2, p1');        % Transform all p1 to p2

% Look at the residual errors of all the points.
dp = (p2map' - p2);
rsq = dp(1,:).^2 + dp(2,:).^2;    % residual squared distance errors

rmsErr = sqrt( sum(rsq)/length(rsq) );
disp('RMS error: '), disp(rmsErr);

return
