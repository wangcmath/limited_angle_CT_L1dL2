%=============================================================
% demo_Synthetic_SL_L1dL2 ----  Limited angle CT recontruction under the
%                                 Gaussian noise model. 
%
% Solves
%           min  norm(x,1)/norm(x,2) + \lambda/2 norm(Au-b,2)^2
%
% Reference: "Limited-Angle CT Reconstruction via the L1/L2  Minimization" 
%             Chao Wang, Min Tao, James Nagy, Yifei Lou 
% Available at: 
%             https://epubs.siam.org/doi/abs/10.1137/20M1341490
% 
% Install the AIR Tools II and IR Tools before running the code
%      AIR Tools II:  https://github.com/jakobsj/AIRToolsII
%      IR Tools: https://github.com/jnagy1/IRtools
%
% Author: Chao Wang  
% Date: June 5 2022
%============================================================= 
close all; clear;
Max_angle = 90;
PRoptions = PRset('angles', 0:Max_angle/30:Max_angle);
[A, btrue, xtrue, ProbInfo] = PRtomo(PRoptions);
sig = 0.005; % 0.5\% noise level 
g = btrue+sig*max(btrue)*randn(size(btrue));

pm.rows = ProbInfo.xSize(1); pm.cols = ProbInfo.xSize(2); 
pm.u_orig = reshape(xtrue, pm.rows, pm.cols);
pm.maxit = 500;
pm.StopCri = 1; pm.tol = 1e-5; % Stopping criterion: relative error between
                           % two consecutive iterations is smaller than
                           % pm.tol; pm.StopCri = 0 turns it off.
pm.box = 1; % without box constaint when pm.box = 0, default setting is with box constraint. 
pm_L1dL2 = pm; 
switch Max_angle
    case 90
        pm_L1dL2.lambda =0.05; pm_L1dL2.beta = .1; pm_L1dL2.rho1 = .1;pm_L1dL2.rho2 = pm_L1dL2.rho1;
        pm_L1mL2.lambda = 5; pm_L1mL2.beta = 10;  pm_L1mL2.rho = 1;
        pm_L1.lambda = 5; pm_L1.beta = 10;  pm_L1.rho = 10;
    case 150
        pm_L1dL2.lambda =0.05; pm_L1dL2.beta = 1; pm_L1dL2.rho1 = 1;pm_L1dL2.rho2 = pm_L1dL2.rho1;
        pm_L1mL2.lambda = 5; pm_L1mL2.beta = 10;  pm_L1mL2.rho = 1;
        pm_L1.lambda = .5; pm_L1.beta = 10;  pm_L1.rho = 10;
end
timestart = tic;
[u_l1dl2,output_l1dl2] = mCTrecon_L1dL2_unconst(A, g, pm_L1dL2);
timeout_l1dl2 = toc(timestart);
figure;
imshow(u_l1dl2)
title(['L1/L2: RE:', num2str(norm(u_l1dl2(:)-xtrue)/norm(xtrue))])