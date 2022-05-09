function [C_bar] = myCbar(Phi,C,C_hat,N)
% MYC_Bar  [C_bar] = myCBar(Phi,C,C_hat.N). 
% X=Phi*x0+Gamma*U             % x=[x1,x2,x3....xN]', U=[u0,u1,u2...uN-1]'
% N is the horizon length. 
% f_k=C_f*x_k+D_f*u_k          % C_hat is the?terminated coefficient matrix   
%   F=Cf_bar*x0 +Df_bar*U;     % F=[f0,f1,f2...fN]'; 
% Cf_bar=[Cf;Cf*A;Cf*A^2;...;Cf_hat*A^N]
% Phi=[A;A^2;A^3;...;A^N];
% Phi_extend=[eye(size(Phi,2));Phi];
CC=[kron(eye(N),C),zeros(N*size(C,1),size(C_hat,2));zeros(size(C_hat,1),N*size(C,2)),C_hat];
C_bar=CC*Phi; 
end
