function [Gamma] = myBbar(A,B,N)
% MYPREDICTION  [Phi,Gamma] = myPrediction(A,B,N). 
% A and B are discrete-time state space matrices for x[k+1]=Ax[k]+Bu[k]
% N is the horizon length. 
%AA*X= gg*X0+   BB*U                PS: X=[x1...xN], U=[u0...uN-1]
%   X=Phi*X0+Gamma*U
n=size(A,1);                        % n is number of x states
m=size(B,2);                         % m is number of u control variables
AA=eye(N*n)-[zeros(n,N*n);kron(eye(N-1),A),zeros((N-1)*n,n)];   %AA
gg=[A;zeros((N-1)*n,n)];            %gg
BB=kron(eye(N),B);                  %BB
% Gamma=AA\BB;                        %Gamma 
Gamma=[zeros(n,N*m);AA\BB];         %Gamma 
% Gamma=[Gamma, zeros((N+1)*n,m)];    %Gamma  
end

