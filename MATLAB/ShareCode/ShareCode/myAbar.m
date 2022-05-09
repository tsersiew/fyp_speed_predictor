function [Phi] = myAbar(A,N)
% MYPREDICTION  [Phi,Gamma] = myPrediction(A,B,N). 
% A and B are discrete-time state space matrices for x[k+1]=Ax[k]+Bu[k]
% N is the horizon length. 
%AA*X= gg*X0+   BB*U                PS: X=[x1...xN], U=[u0...uN-1]
%   X=Phi*X0+Gamma*U
n=size(A,1);                        % n is number of x states
AA=eye(N*n)-[zeros(n,N*n);kron(eye(N-1),A),zeros((N-1)*n,n)];   %AA
gg=[A;zeros((N-1)*n,n)];            %gg
% Phi=AA\gg;                          %Phi
Phi=[eye(n);AA\gg];                %Phi
end


