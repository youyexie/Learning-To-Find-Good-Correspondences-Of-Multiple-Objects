function [fSuccess, H_m_c, indicesBest, rmsErr] = ...
    sub_Pose_from_2D_3D_Ransac(p2d, p3d, K, W,H, sigmas)
% Compute the pose of the camera wrt the world from the given
% putative 2D to 3D point correspondences; also eliminate
% outliers.  If no solution is found, H is returned null.
%
% Input parameters:
%   p2d - image points in pixels, size is (3,N)
%   p3d - correspond 3D world points, size is (4,N)
%   K - intrinsic camera parameter matrix (3,3)
%   W,H, - width and height of image
%   sigmas - uncertainties of image points, size 2xN
% Output:
%   fSuccess:   set to true if we are successful
%   pose_m_c:   pose of model to camera
%   indicesBest - the indices of the inliers
%   rmsErr - RMS reprojection error (pixels) of inliers
%
% Will use the MSAC (m-estimator sample consensus) algorithm to find
% inliers.  See the paper: "MLESAC: A new robust estimator with application
% to estimating image geometry" by Torr and Zisserman, at
% http://www.robots.ox.ac.uk/~vgg/publications/2000/Torr00/torr00.pdf

% Set all output parameters to something
fSuccess = false;
H_m_c = [];
rmsErr = [];
indicesBest = [];

% Make sure we have the same number of points in both sets
N = size(p2d,2);    % Number of corresponding point pairs
assert(N == size(p3d,2));

% Define the minimum number of inliers we need.
minNumInliers = 5;
if N<minNumInliers  return;  end

% Now, if a point is an outlier, its residual error can be anything; i.e.,
% any value between 0 and the size of the image.  Let's
% assume a uniform probability distribution.
pOutlier = 1/(W*H);

% Estimate the fraction of inliers.  We'll actually estimate this later
% from the data, but start with a worst case scenario assumption.
inlierFraction = 0.1;

% Let's say that we want to get a good sample with probability Ps.
Ps = 0.99;

% Determine the number of iterations needed (this will change).
nInlier = round(N*inlierFraction); % number of inliers among the N points
m = nInlier*(nInlier-1)*(nInlier-2);    % #ways to choose 3 inliers
m = max(m,1);   % Make sure m is at least 1
n = N*(N-1)*(N-2);  % #ways to choose 3 points
p = m/n;        % probability that any sample of 3 points is all inliers
nIterations = log(1-Ps) / log(1 - p);
nIterations = ceil(nIterations);
%fprintf('Initial estimated number of iterations needed: %d\n', nIterations);

MAXITERATIONS = 1000;   % Never go above this many iterations

% Do RANSAC iterations.
% At each iteration draw a random sample of 3 point pairs and estimate the
% pose transformation between the two sets of points.
sample_count = 0;   % The number of Ransac trials
pBest = -Inf;       % Best probability found so far (actually, the log)
while sample_count < nIterations
    if sample_count > MAXITERATIONS
        return;
    end
    
    % Grab 3 matching points at random from p2d, p3d
    v = randperm(N);
    ps2d = p2d(:,v(1:3));
    ps3d = p3d(:,v(1:3));
    
    %fprintf('iteration %d:  indices ', sample_count);
    %fprintf('%d %d %d\n', v(1),v(2),v(3));
    
    % Estimate a pose using those points.  There are up to four possible
    % solutions.
    H_all = p3p(ps2d, ps3d, K);
    if isempty(H_all)
        %fprintf('No solutions found\n');
        continue;   % No solutions were found
    end
    
    % Ok, we were able to determine a pose with this sample.
    sample_count = sample_count + 1;
    
    % For each possible solution, estimate the error residuals for all points
    for iSoln=1:size(H_all,3)
        H_m_c = H_all(:,:,iSoln);   % Pose, model to camera
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
        nInlier = sum(indicesInliers);
        if nInlier < 3
            continue;   % bad solution
        end
        
        % Update the number of iterations required
        if nInlier/N > inlierFraction
            inlierFraction = nInlier/N;
            
            m = nInlier*(nInlier-1)*(nInlier-2);
            n = N*(N-1)*(N-2);  % #ways to choose 3 points
            p = m/n;  % probability that any sample of 3 points is all inliers
            nIterations = log(1-Ps) / log(1 - p);
            nIterations = ceil(nIterations);
            %fprintf('New estimated number of iterations needed: %d\n', nIterations);
        end
        
        % Net probability of the data (log)
        p = sum(log(pInlier(indicesInliers))) + (N-nInlier)*log(pOutlier);
        
        % Keep this solution if probability is better than the best so far.
        if p>pBest
            pBest = p;
            HBest = H_m_c;
            indicesBest = indicesInliers;
            %fprintf(' best so far with %d inliers, p=%f\n', nInlier, p);
            %disp(find(indicesBest));
        end
    end  % end for iSoln=1...
end
fprintf('Number of RANSAC samples tried: %d\n', sample_count);

if isempty(indicesBest)
    return;     % Couldn't find a solution
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ok, recompute the pose using all the inliers.
p2dInliers = p2d(:,indicesBest);
p3dInliers = p3d(:,indicesBest);

% Put 2D points into a column vector
y0 = reshape(p2dInliers(1:2,:), [], 1);

% We'll use an iterative least squares method.  We have a good starting
% guess in HBest. Get the initial guess of the pose in the form [wx wy wz
% tx ty tz], where wx,wy,wz is the rotation axis multiplied by the rotation
% angle, and tx,ty,tz is the translation.
th = acos( (HBest(1,1)+HBest(2,2)+HBest(3,3)-1)/2 );
if abs(th)<1e-6
    % No rotation; axis is undefined
    kx = 1;     ky = 0;     kz = 0;
else
    kx = (HBest(3,2)-HBest(2,3))/(2*sin(th));
    ky = (HBest(1,3)-HBest(3,1))/(2*sin(th));
    kz = (HBest(2,1)-HBest(1,2))/(2*sin(th));
end
wx = kx*th;
wy = ky*th;
wz = kz*th;
tx = HBest(1,4);
ty = HBest(2,4);
tz = HBest(3,4);

x = [wx; wy; wz; tx; ty; tz];

for i=1:10
    % Get predicted image points
    y = fProject(x, p3dInliers, K);
    
    % Estimate Jacobian
    e = 0.00001;    % a tiny number
    J(:,1) = ( fProject(x+[e;0;0;0;0;0],p3dInliers,K) - y )/e;
    J(:,2) = ( fProject(x+[0;e;0;0;0;0],p3dInliers,K) - y )/e;
    J(:,3) = ( fProject(x+[0;0;e;0;0;0],p3dInliers,K) - y )/e;
    J(:,4) = ( fProject(x+[0;0;0;e;0;0],p3dInliers,K) - y )/e;
    J(:,5) = ( fProject(x+[0;0;0;0;e;0],p3dInliers,K) - y )/e;
    J(:,6) = ( fProject(x+[0;0;0;0;0;e],p3dInliers,K) - y )/e;
    
    % Error is observed image points - predicted image points
    dy = y0 - y;
    
    % Ok, now we have a system of linear equations   dy = J dx
    % Solve for dx using the pseudo inverse
    dx = pinv(J) * dy;
    
    % Stop if parameters are no longer changing
    if abs( norm(dx)/norm(x) ) < 1e-6
        break;
    end
    
    x = x + dx;   % Update pose estimate
end


% Get resulting pose params
wx = x(1); wy = x(2); wz = x(3);
tx = x(4); ty = x(5); tz = x(6);

th = norm([wx;wy;wz]);  % Angle
if abs(th)<1e-6
    % No rotation; axis is undefined
    kx = 1;     ky = 0;     kz = 0;
else
    kx = wx/th;
    ky = wy/th;
    kz = wz/th;
end
c = cos(th);
s = sin(th);
v = 1 - cos(th);

% Rotation matrix, model to camera
R = [ kx*kx*v+c     kx*ky*v-kz*s   kx*kz*v+ky*s ;
    kx*ky*v+kz*s  ky*ky*v+c      ky*kz*v-kx*s ;
    kx*kz*v-ky*s  ky*kz*v+kx*s   kz*kz*v+c ];

H_m_c = [ R  [tx;ty;tz]; 0 0 0 1 ];
% w = x(1:3);
% t = x(4:6);
% pose_m_c = class_Pose(w,t);

% Compute final RMS error.  dy is the vector of residuals.
rmsErr = sqrt( sum(dy.^2)/size(p2dInliers,2) );

fSuccess = true;

return



function p = fProject(x, P_M, K)
% Project 3D points onto image

% Get pose params
wx = x(1); wy = x(2); wz = x(3);
tx = x(4); ty = x(5); tz = x(6);

th = norm([wx;wy;wz]);  % Angle
if abs(th)<1e-6
    % No rotation; axis is undefined
    kx = 1;     ky = 0;     kz = 0;
else
    kx = wx/th;
    ky = wy/th;
    kz = wz/th;
end
c = cos(th);
s = sin(th);
v = 1 - cos(th);

% Rotation matrix, model to camera
R = [ kx*kx*v+c     kx*ky*v-kz*s   kx*kz*v+ky*s ;
    kx*ky*v+kz*s  ky*ky*v+c      ky*kz*v-kx*s ;
    kx*kz*v-ky*s  ky*kz*v+kx*s   kz*kz*v+c ];

% Extrinsic camera matrix
Mext = [ R  [tx;ty;tz] ];

% Project points
ph = K*Mext*P_M;

% Divide through 3rd element of each column
ph(1,:) = ph(1,:)./ph(3,:);
ph(2,:) = ph(2,:)./ph(3,:);
ph = ph(1:2,:); % Get rid of 3rd row

p = reshape(ph, [], 1); % reshape into 2Nx1 vector
return