classdef class_Pose < handle
    % Pose class
    
    properties (SetAccess = private, GetAccess = private)
        Hmatrix     % 4x4 transformation matrix
        Hmatrix_inv     % The inverse of Hmatrix
        
        % Translation (a 3x1 vector)
        trans       % tx;ty;tz
        
        % Rotation represented as XYZ fixed angles in radians
        avector     % 3x1 vector (ax;ay;az)
        
        % Rotation represented as angle*axis
        wvector     % 3x1 vector (wx;wy;wz)
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constructor
        %   Create a pose object and initialize it via either:
        %       pose = class_Pose(H);
        %       pose = class_Pose(w,t);
        function obj = class_Pose(varargin)
            if nargin == 1
                obj.Hmatrix = varargin{1};
            elseif nargin == 2
                obj.wvector = varargin{1};
                obj.trans = varargin{2};
            else
                disp('error in constructing pose');
            end
        end    % end constructor
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Set the pose to the desired value.  Use this when you don't want
        % to create a new pose object; just modify the old one.
        function setPose(obj, varargin)
            if nargin == 1
                obj.Hmatrix = varargin{1};
                
                % These are no longer valid
                obj.Hmatrix_inv = [];
                obj.trans = [];
                obj.wvector = [];
                obj.avector = [];
            elseif nargin == 2
                obj.wvector = varargin{1};
                obj.trans = varargin{2};
                
                % These are no longer valid
                obj.Hmatrix = [];
                obj.Hmatrix_inv = [];
                obj.avector = [];
            else
                disp('error in setting pose');
            end
        end     % end function setPose
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Copy the pose.
        function pose = copy(obj)
            if isempty(obj.wvector)
                pose = class_Pose(obj.Hmatrix);
            else
                pose = class_Pose(obj.wvector, obj.trans);
            end
        end     % end function copy
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get H matrix
        function H = H(obj)
            % Check if we need to create the 4x4 homogeneous matrix
            if isempty(obj.Hmatrix)
                % Assume that the w,t representation exists.  First find
                % the rotation matrix.
                wx = obj.wvector(1);  wy = obj.wvector(2);  wz = obj.wvector(3);
                
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
                
                % Create rotation matrix
                R = [ kx*kx*v+c     kx*ky*v-kz*s   kx*kz*v+ky*s ;
                    kx*ky*v+kz*s  ky*ky*v+c      ky*kz*v-kx*s ;
                    kx*kz*v-ky*s  ky*kz*v+kx*s   kz*kz*v+c ];
                
                T = obj.trans;   % translation vector
                
                H = [ ...
                    R   T;
                    0 0 0 1 ];
                obj.Hmatrix = H;    % Save in case we use later
            else
                H = obj.Hmatrix;
            end
        end   % end function H
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get the rotation matrix
        function R = R(obj)
            H = obj.H;
            R = H(1:3,1:3);
        end   % end function R      
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get w vector
        function w = w(obj)
            % Check if we need to create the 3x1 vector (wx,wy,wz),
            % which is the rotation angle in radians times the rotation
            % unit vector.
            if isempty(obj.wvector)
                % Assume that the H representation exists.
                R = obj.Hmatrix(1:3,1:3);   % Get rotation matrix
                th = acos( (R(1,1)+R(2,2)+R(3,3)-1)/2 );
                if abs(th)<1e-6
                    % No rotation; axis is undefined
                    kx = 1;     ky = 0;     kz = 0;
                else
                    kx = (R(3,2)-R(2,3))/(2*sin(th));
                    ky = (R(1,3)-R(3,1))/(2*sin(th));
                    kz = (R(2,1)-R(1,2))/(2*sin(th));
                end
                wx = kx*th;
                wy = ky*th;
                wz = kz*th;
                
                obj.wvector = [wx;wy;wz];
            end
            w = obj.wvector;
        end   % end function w
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get the "a" vector
        function a = a(obj)
            % Check if we need to create the 3x1 vector (ax,ay,az),
            % where (ax,ay,az) are the XYZ fixed angles in radians
            if isempty(obj.avector)
                H = obj.H;
                ay = atan2( -H(3,1), sqrt( H(1,1)^2 + H(2,1)^2 ) );
                cy = cos(ay);
                
                if abs(cy) > 1e-10          % some tiny number
                    az = atan2( H(2,1)/cy, H(1,1)/cy );
                    ax = atan2( H(3,2)/cy, H(3,3)/cy );
                else
                    % We have a degenerate solution
                    az = 0;
                    ax = atan2(H(1,2),H(2,2));
                    if ay<0     ax = -ax;   end
                end
                
                obj.avector = [ ax; ay; az ];
            end
            a = obj.avector;
        end   % end function a
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get t (translation)
        function t = t(obj)
            if isempty(obj.trans)
                obj.trans = obj.Hmatrix(1:3,4);
            end
            t = obj.trans;
        end   % end function t
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Invert the pose (create a new pose that is the inverse)
        %  (This overloads the inv() function for this class)
        function p2 = inv(p1)
            H = Hinv(p1);
            p2 = class_Pose(H);
        end     % end function inv
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Get the inverse of the H matrix.
        function Hinv = Hinv(obj)
            if isempty(obj.Hmatrix_inv)
                Hinv = inv(obj.H);
                obj.Hmatrix_inv = Hinv;    % Save in case we use later
            else
                Hinv = obj.Hmatrix_inv;
            end
        end     % end function Hinv
             
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Combine the two poses by multiplying their H matrices.
        %  (This overloads the * operator for this class)
        function p = mtimes(p1, p2)
            H = p1.H * p2.H;
            p = class_Pose(H);
        end     % end function mtimes
        
    end     % end methods
    
end     % end classdef

