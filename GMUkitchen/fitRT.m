function [R,T] = fitRT(p1_3d,p2,K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p1_3d   : 3d points from image 1, 3xN
% p2      : 2d points from image 2, 2xN
% K       ? intrinsic matrix
% (R*p1_3d+T)/z = normalized_p2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% normalized the 2d point
p2 = p2(1:2,:);
N = size(p2,2);
nor_p2 = inv(K)*[p2;ones(1,N)];



A = zeros(N,12);
for i=1:N
    X = p1_3d(1,i); Y = p1_3d(2,i); Z = p1_3d(3,i);
    x = nor_p2(1,i); y = nor_p2(2,i);
    A( 2*(i-1)+1, :) = [ X Y Z 0 0 0 -x*X -x*Y -x*Z 1 0 -x ];
    A( 2*(i-1)+2, :) = [ 0 0 0 X Y Z -y*X -y*Y -y*Z 0 1 -y ];
end

% Solve for the value of x that satisfies Ax = 0.
% The solution to Ax=0 is the singular vector of A corresponding to the
% smallest singular value; that is, the last column of V in A=UDV'
[U,D,V] = svd(A);
x = V(:,end); % get last column of V
% Reshape x back to a 3x4 matrix, M = [R t]
M = [ x(1) x(2) x(3) x(10);
x(4) x(5) x(6) x(11);
x(7) x(8) x(9) x(12) ];

R = M(1:3,1:3);
T = M(:,4);


end