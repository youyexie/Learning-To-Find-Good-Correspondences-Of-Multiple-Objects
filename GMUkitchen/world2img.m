
% Transform 3D world coordinates to image coordinates for the GMU Kitchen scenes dataset.
% Georgios Georgakis, 2016

% INPUT 
%   points: 3d world points (M x 3).
%   colors: RGB values for the 3d world points (M x 3).
%   cam1: camera pose information for the chosen frame. This is equal to
%           frames(i) for a given frame i, which can be loaded from 
%           gmu_scene_00#_reconstruct_info_frame_sort.mat
%   kinectParams: Struct that holds focal length and cx, cy from the Kinect
%           calibration. The values can be found in scene_pose_info/calib_color.yaml
%           fx_rgb = 1.0477637710998533e+03;
%           fy_rgb = 1.0511749325842486e+03;
%           cx_rgb = 9.5926120509632392e+02;
%           cy_rgb = 5.2911546499433564e+02;
%   imsize: frame dimensions [imh, imw]=size(img)

% OUTPUT
%   points2D: 2D image image
%   col_proj: corresponding RGB colors for the image points

% Note that the points and colors can be loaded when the 3d scene point cloud
% is read with readPly()

function [points2D] = world2img(points, cam1, imsize)

imh=imsize(1); imw=imsize(2);
Rw2c=cam1.Rw2c; Tw2c=cam1.Tw2c;

% transform from world to camera frame
camera_pcl = bsxfun(@plus, Tw2c', Rw2c*points'); % cam = R*world + T
% project on image coordinates
X=camera_pcl(1,:); Y=camera_pcl(2,:); Z=camera_pcl(3,:);
x_proj = round((-X.*1.0477637710998533e+03./Z) + 9.5926120509632392e+02);
y_proj = round((Y.*1.0511749325842486e+03./Z) + 5.2911546499433564e+02);

% keep only the coordinates in the image frame
valid=find(~isnan(x_proj) & ~isnan(y_proj) & x_proj>0 & y_proj>0 & x_proj<=imw & y_proj<=imh);
x_proj=round(x_proj(valid)); y_proj=round(y_proj(valid));
%col_proj = colors(valid,:);
points2D=[x_proj ; y_proj];

