function H = p3p(p2d, p3d, K)
% Estimate a pose using 3 point correspondences.
% This method is described in Fischler's RANSAC paper.
% Inputs:  
%   p2d: 3 columns, each column is (x,y,1)
%   p3d: 3 columns, each column ix (X,Y,Z,1)
%   K:  camera intrinsic matrix (3x3)
% Outputs:
%   H:  a matrix of size (4,4,n).  Each 4x4 submatrix is a possible pose
%   solution, of model-to-camera.  There are up to n=4 solutions.
%   If no solution is found, return H = [].

H = [];

if size(p2d,2) ~= 3 || size(p3d,2) ~= 3
    error('Need exactly 3 points as input to function p3p');
end
if size(p2d,1) ~= 3
    error('Function p3p expects homogeneous image points as input');
end

% Make sure points are not colinear (also checks if two points are
% coincident).  Points are colinear if  (p3-p1)x(p2-p1) = 0
if norm(cross(p2d(:,3)-p2d(:,1), p2d(:,2)-p2d(:,1))) < 1e-6
    %disp('Image points are degenerate:'), disp(p2d);    
    return;
end
if norm(cross(p3d(1:3,3)-p3d(1:3,1), p3d(1:3,2)-p3d(1:3,1))) < 1e-6
    %disp('Model points are degenerate:'), disp(p3d);    
    return;
end

Kinv = inv(K);

% Normalized image points
pn = Kinv * p2d;

% Compute unit vectors in direction of observed points
for i=1:3
    q(:,i) = pn(:,i)/norm(pn(:,i));
end

% Compute distances between model points
d12 = norm( p3d(:,1) - p3d(:,2) );
d13 = norm( p3d(:,1) - p3d(:,3) );
d23 = norm( p3d(:,2) - p3d(:,3) );

% Compute dot products between image unit vectors
q12 = dot(q(:,1), q(:,2));
q13 = dot(q(:,1), q(:,3));
q23 = dot(q(:,2), q(:,3));

K1 = (d23/d13)^2;
K2 = (d23/d12)^2;

G4 = (K1*K2-K1-K2)^2 - 4*K1*K2*q23^2;
G3 = 4*(K1*K2-K1-K2)*K2*(1-K1)*q12 + 4*K1*q23*((K1*K2+K2-K1)*q13 + 2*K2*q12*q23);
G2 = (2*K2*(1-K1)*q12)^2 + 2*(K1*K2+K1-K2)*(K1*K2-K1-K2) + ...
    4*K1*((K1-K2)*q23^2 + (1-K2)*K1*q13^2 - 2*K2*(1+K1)*q12*q13*q23);
G1 = 4*(K1*K2+K1-K2)*K2*(1-K1)*q12 + 4*K1*((K1*K2-K1+K2)*q13*q23 + 2*K1*K2*q12*q13^2);
G0 = (K1*K2+K1-K2)^2 - 4*K1^2*K2*q13^2;

% k1 can be any of the positive real roots of the quadradic
% [G4*(k1^4)+G3*(k1^3)+G2*(k1^2)+G1*(k1)+G0]
xroot = roots([G4 G3 G2 G1 G0]);

% Keep positive real roots only
x = [];
for i=1:length(xroot)
    if(isreal(xroot(i))&& xroot(i)>0)
        x = [x xroot(i)];
    end
end
if length(x)==0
    return;     % no solutions were found
end

% Solve for a,b,c's
for i=1:length(x)
    a(i) = d12/sqrt(x(i)^2 - 2*x(i)*q12 + 1);
    b(i) = a(i)*x(i);

    m = 1-K1;
    n = 2*(K1*q13-x(i)*q23);
    p = x(i)^2-K1;

    r = 1;
    s = 2*(-x(i)*q23);
    t = x(i)^2*(1-K2)+2*x(i)*K2*q12-K2;

    if(m*t == r*p)
        y(1,i) = q13 + sqrt(q13^2+(d13^2 - a(i)^2)/a(i)^2);
        y(2,i) = q13 - sqrt(q13^2+(d13^2 - a(i)^2)/a(i)^2);
    else
        y(1,i) = (s*p - n*t)/(m*t - r*p);
    end
end

% Build abc vectors
abc = [];
for i=1:length(x)
    if(size(y,1)==2 && y(2,i)~=0)
        abc_i = [a(i) b(i) (a(i)*y(1,i))]';
        abc_ii = [a(i) b(i) (a(i)*y(2,i))]';
        abc = [abc abc_i abc_ii];
    else
        abc_i = [a(i) b(i) (a(i)*y(1,i))]';
        abc = [abc abc_i];
    end
end

% Check abc sets, Law of Cosines
isvalid = zeros(1,size(abc,2));
th = .01;
for i=1:size(abc,2)
    d12_i = sqrt(abc(1,i)^2 + abc(2,i)^2 - 2*abc(1,i)*abc(2,i)*q12);
    d13_i = sqrt(abc(1,i)^2 + abc(3,i)^2 - 2*abc(1,i)*abc(3,i)*q13);
    d23_i = sqrt(abc(2,i)^2 + abc(3,i)^2 - 2*abc(2,i)*abc(3,i)*q23);

    if(abs(d12_i-d12)<th && abs(d13_i-d13)<th && abs(d23_i-d23)<th)
        isvalid(i) = 1;
    end
end

if(sum(isvalid)~=length(isvalid))
    %fprintf('\nOne of more tetrahedron solutions invalid!!!\n');
    return;
end

nSolns = 0;
for i=1:size(abc,2)
    if(isvalid(i))
        nSolns = nSolns+1;
        P_cam(1:3,1,nSolns) = abc(1,i)*q(:,1);
        P_cam(1:3,2,nSolns) = abc(2,i)*q(:,2);
        P_cam(1:3,3,nSolns) = abc(3,i)*q(:,3);
        P_cam(4,:,nSolns) = 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find relative pose between the two sets of 3D points.
% The approach is to align two triangles, one step at a time.
% This solution is described in the Shapiro and Stockman book, p. 419.

for i=1:size(P_cam, 3)
    % Construct translation matrices to shift the first point in each set to the
    % origin.
    T1 = [ eye(3)  -P_cam(1:3,1,i);  0 0 0 1 ];
    T2 = [ eye(3)  -p3d(1:3,1);  0 0 0 1 ];

    % Construct rotation matrices so that vector 1->2 lies along the x axis
    % This is equivalent to a rotation about the axis perpendicular to the
    % x-axis and vector 1->2
    d12 = P_cam(1:3,2,i)-P_cam(1:3,1,i);
    crossx12 = cross( d12, [1;0;0] );
    if norm(crossx12) == 0
        % Vector 1->2 already lies along the x axis
        R1 = eye(4);
    else
        k = crossx12 / norm(crossx12);                 % unit vector
        a = acos( dot( d12, [1;0;0] ) / norm(d12) );   % angle
        ca = cos(a);  sa = sin(a);   va = 1-ca;
        R1 = [ k(1)*k(1)*va+ca   k(1)*k(2)*va-k(3)*sa   k(1)*k(3)*va+k(2)*sa   0;
            k(1)*k(2)*va+k(3)*sa   k(2)*k(2)*va+ca   k(2)*k(3)*va-k(1)*sa   0;
            k(1)*k(3)*va-k(2)*sa   k(2)*k(3)*va+k(1)*sa   k(3)*k(3)*va+ca   0;
            0 0 0 1];
    end

    d12 = p3d(1:3,2)-p3d(1:3,1);
    crossx12 = cross( d12, [1;0;0] );
    if norm(crossx12) == 0
        % Vector 1->2 already lies along the x axis
        R2 = eye(4);
    else
        k = crossx12 / norm(crossx12);                 % unit vector
        a = acos( dot( d12, [1;0;0] ) / norm(d12) );   % angle
        ca = cos(a);  sa = sin(a);   va = 1-ca;
        R2 = [ k(1)*k(1)*va+ca   k(1)*k(2)*va-k(3)*sa   k(1)*k(3)*va+k(2)*sa   0;
            k(1)*k(2)*va+k(3)*sa   k(2)*k(2)*va+ca   k(2)*k(3)*va-k(1)*sa   0;
            k(1)*k(3)*va-k(2)*sa   k(2)*k(3)*va+k(1)*sa   k(3)*k(3)*va+ca   0;
            0 0 0 1];
    end

    % Construct a rotation about the x axis so that point 3 lies in the xy
    % plane. This is equivalent to a rotation that makes the vector
    % perpendicular to 1->3 and the x-axis, point in the z direction.
    d13 = R1*T1*(P_cam(:,3,i)-P_cam(:,1,i));
    d13 = d13(1:3);

    % projection of d13 onto xy plane
    p13 = [ 0; d13(2); d13(3) ];
    crossx13 = cross( p13,[0;1;0] );
    dot13 = dot( p13,[0;1;0] );

    if norm(crossx13) == 0
        % Vector 1->3 already lies in xy plane
        R3 = eye(4);
    else
        a = acos( dot13/norm(p13) );
        k = crossx13 / norm(crossx13);
        Rx = [1, 0, 0; 0, cos(a), -k(1)*sin(a); 0, k(1)*sin(a), cos(a)];
        R3 = [[Rx [0 0 0]']; 0 0 0 1];
    end

    d13 = R2*T2*(p3d(:,3)-p3d(:,1));
    d13 = d13(1:3);

    % projection of d13 onto xy plane
    p13 = [ 0; d13(2); d13(3) ];
    crossx13 = cross( p13,[0;1;0] );
    dot13 = dot( p13, [0;1;0] );

    if norm(crossx13) == 0
        % Vector 1->3 already lies in xy plane
        R4 = eye(4);
    else
        a = acos( dot13/norm(p13) );
        k = crossx13 / norm(crossx13);

        Rx = [1, 0, 0; 0, cos(a), -k(1)*sin(a); 0, k(1)*sin(a), cos(a)];
        R4 = [[Rx [0 0 0]']; 0 0 0 1];
    end

    % The two sets of points are aligned, ie, R3*R1*T1*P_cam = R4*R2*T2*P_M
    H(:,:,i) = inv(T1)*inv(R1)*inv(R3)*R4*R2*T2;
end

return
