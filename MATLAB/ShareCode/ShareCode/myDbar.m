function [D_bar] = myDbar(Gamma,C,C_hat,D,N)
% MYC_Bar  [C_bar] = myCBar(Phi,C,C_hat.N). 
% X=Phi*x0+Gamma*U             % x=[x1,x2,x3....xN]', U=[u0,u1,u2...uN-1]'
% N is the horizon length. 
% f_k=C_f*x_k+D_f*u_k          % C_hat is the?terminated coefficient matrix   
%   F=Cf_bar*x0 +Df_bar*U;     % F=[f0,f1,f2...fN]'; 
% Cf_bar=[Cf;Cf*A;Cf*A^2;...;Cf_hat*A^N]
CC=[kron(eye(N),C),zeros((N)*size(C,1),size(C_hat,2));zeros(size(C_hat,1),(N)*size(C,2)),C_hat];
D_bar=[kron(eye(N),D);zeros(size(C,1),N*size(D,2))];
D_bar=D_bar+CC*Gamma;
end
