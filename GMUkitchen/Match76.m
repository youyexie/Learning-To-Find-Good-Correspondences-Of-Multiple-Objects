clear all;close all;clc


% import an depth image
img = imread('gmu_scene_007\Images\rgb_743.png');
depth = imread('gmu_scene_007\Depths\depth_743.png');
scale = 5.8562;
load('gmu-kitchens_info\scene_pose_info\scene_7_reconstruct_info_frame_sort.mat');
cam1 = frames(729)

% intrinsic matrix
K = [  1.0477637710998533e+03 0                       9.5926120509632392e+02;
       0                      1.0511749325842486e+03  5.2911546499433564e+02;
       0                      0                       1];

% depth to 3d point
[cam_point3D, wrold_point3D] = depth2world(depth, scale, cam1);

% 3d point to image
[h,w,d] = size(img)
imsize = [h,w]

[point2D] = world2img(wrold_point3D', cam1, imsize); %[x,y]
[ temp , num_point ]= size(point2D);

normalized_cam_point3D = inv(K)*[point2D;ones(1,num_point)];

% generate the corrsponding 3D cloud image
CloudImg = zeros(h,w,d);


for i =1:num_point
    CloudImg(point2D(2,i),point2D(1,i),:) = cam_point3D(:,i);
end

% positive of x-axis to the right 
CloudImg(:,:,1) = -CloudImg(:,:,1); 
%%%
figure(1);
imshow(img)

figure(2)
imshow(CloudImg(:,:,3),[])
impixelinfo

%%% first image

run('vlfeat-0.9.21-bin\vlfeat-0.9.21\toolbox\vl_setup')
I1 = imread('img\s7_rgb_743.png');
%I1 = single(I1); % Convert to single precision floating point
if size(I1,3)>1; I1 = rgb2gray(I1); end
I1 = single(I1);
figure(3);
imshow(I1,[]);
% These parameters limit the number of features detected
peak_thresh = 0; % increase to limit; default is 0
edge_thresh = 10; % decrease to limit; default is 10
[f1,d1] = vl_sift(I1, ...
'PeakThresh', peak_thresh, ...
'edgethresh', edge_thresh );
fprintf('Number of frames (features) detected: %d\n', size(f1,2));
% Show all SIFT features detected
fig = vl_plotframe(f1) ;
set(fig,'color','y','linewidth',2) ;

feature_mask = zeros(h,w);

for i =1:size(f1,2)
    feature_mask(round(f1(2,i)),round(f1(1,i))) = 1;
end

% filter out the point with no 3D point
indices = [];
for i =1:size(f1,2)
    
    if CloudImg(round(f1(2,i)),round(f1(1,i)))==0
        indices = [indices i];   
    end
end
f1(:,indices)=[];
d1(:,indices)=[];


figure(31);
imshow(I1,[]);
fig = vl_plotframe(f1) ;
set(fig,'color','y','linewidth',2);
    

%%% second image
I2 = imread('img\s6_rgb_315.png');
figure;imshow(I2,[])
if size(I2,3)>1 I2 = rgb2gray(I2); end
I2 = single(I2);
figure, imshow(I2,[]);
% These parameters limit the number of features detected
peak_thresh = 0; % increase to limit; default is 0
edge_thresh = 10; % decrease to limit; default is 10
[f2,d2] = vl_sift(I2, ...
'PeakThresh', peak_thresh, ...
'edgethresh', edge_thresh );
fprintf('Number of frames (features) detected: %d\n', size(f2,2));
% Show all SIFT features detected
h = vl_plotframe(f2) ;
set(h,'color','y','linewidth',2) ;

%
% Threshold for matching
% Descriptor D1 is matched to a descriptor D2 only if the distance d(D1,D2)
% multiplied by THRESH is not greater than the distance of D1 to all other
% descriptors
thresh = 1.5; % default = 1.5; increase to limit matches
[matches, scores] = vl_ubcmatch(d1, d2, thresh);
fprintf('Number of matching frames (features): %d\n', size(matches,2));
indices1 = matches(1,:); % Get matching features
f1match = f1(:,indices1);
d1match = d1(:,indices1);
indices2 = matches(2,:);
f2match = f2(:,indices2);
d2match = d2(:,indices2);


% Show matches
figure, imshow([I1;I2],[]);
o = size(I1,1) ;
line([f1match(1,:);f2match(1,:)], ...
[f1match(2,:);f2match(2,:)+o]) ;
for i=1:size(f1match,2)
x = f1match(1,i);
y = f1match(2,i);
text(x,y,sprintf('%d',i), 'Color', 'r');
end
for i=1:size(f2match,2)
x = f2match(1,i);
y = f2match(2,i);
text(x,y+o,sprintf('%d',i), 'Color', 'r');
end


%%% get the maching points 2D point and 3D point

num_match = size(f1match ,2);

point_3d = zeros(3,num_match);
point_2d = zeros(2,num_match);
normalized_point_2d = zeros(2,num_match);

for i = 1:num_match
    point_3d(:,i) = CloudImg(round(f1match(2,i)),round(f1match(1,i)),:);
end

for i = 1:num_match
    point_2d(:,i) = f2match(1:2,i);
end

normalized_point_2d = inv(K)*[point_2d;ones(1,num_match)];
normalized_point_2d = normalized_point_2d(1:2,:);

%
input = [point_3d;normalized_point_2d]';
%%%
save('FaceClassifier\InputData\\match76.mat','input')

%% load the inference result

result = readNPY('FaceClassifier\InferenceResult\match76.npy');

%% the inference result for objects

PostProcessingRT(point_3d,K,f1match,f2match,I1,I2,result,1.0)