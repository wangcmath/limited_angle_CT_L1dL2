close all; clear;
K = [0.05 0.05];
window = ones(8);
L = 1;
%=============================================================
% demo_noiseless ---- Solve a limited CT reconstruction via L1/L2 (noiseless case)
%
% Solves
%           min  norm(x,1)/norm(x,2)
%           s.t. Ax = b, p<=x<=q
%
% Reference: "Minimizing L 1 over L 2 norms on the gradient" 
%             Chao Wang, Min Tao, Chen-Nee Chuah, James G Nagy, Yifei Lou 
% Available at: 
%             https://iopscience.iop.org/article/10.1088/1361-6420/ac64fb/
% 
% Author: Chao Wang  
% Date: Jun. 5 2022
%============================================================= 
angle = 45;

    Max_angle = angle; % 30 or 180 
    PRoptions = PRset('angles', 0:Max_angle/30:Max_angle);
    [A, btrue, xtrue, ProbInfo] = PRtomo(PRoptions);

    tic
    %%  Parmeter setting
    pm.rows = ProbInfo.xSize(1); pm.cols = ProbInfo.xSize(2); 
    pm.u_orig = reshape(xtrue, pm.rows, pm.cols);
    pm.maxit = 500;
    pm.StopCri = 1; pm.tol = 1e-5; % Stopping criterion: relative error between
                                   % two consecutive iterations is smaller than
                                   % pm.tol; pm.StopCri = 0 turns it off.
    pm.box = 1; % without box constaint when pm.box = 0, default setting is with box constraint. 
    filename = ['SL' num2str(Max_angle) 'noiseless'];

    pm_L1dL2 = pm; pm_L1 = pm;pm_L1mL2 = pm;
    switch Max_angle
        case 30
            pm_L1dL2.lambda =0.05;pm_L1dL2.rho1 = 1;pm_L1dL2.rho2 = 1;pm_L1dL2.beta = 1; 
            pm_L1.rho = 10; pm_L1.lambda = 0.5; pm_L1.beta = 10; 
            pm_L1mL2.rho = 10; pm_L1mL2.lambda = 0.5; pm_L1mL2.beta = 10; 
        case 45
          pm_L1.rho = 10; pm_L1.lambda = 0.5; pm_L1.beta = 50; % for Lp
            pm_L1dL2.lambda =0.05;pm_L1dL2.rho1 = 1;pm_L1dL2.rho2 = 1;pm_L1dL2.beta = 1; % 1.733e-02
            pm_L1mL2.rho = 10; pm_L1mL2.lambda = 0.5; pm_L1mL2.beta = 10; 
        case 60
            pm_L1.rho = 10; pm_L1.lambda = 0.5; pm_L1.beta = 100; % Lp
            pm_L1dL2.lambda =0.05;pm_L1dL2.rho1 = 1;pm_L1dL2.rho2 = 1;pm_L1dL2.beta = .5; 
            pm_L1mL2.rho = 10; pm_L1mL2.lambda = 0.5; pm_L1mL2.beta = 10; 
        case 90
            pm_L1dL2.lambda =0.05;pm_L1dL2.rho1 = 1;pm_L1dL2.rho2 = .1;pm_L1dL2.beta = 1; % 0.05;1;0.1
            pm_L1.rho = 10; pm_L1.lambda = 0.5; pm_L1.beta = 1;     
        case 180
            pm_L1dL2.lambda =.5;pm_L1dL2.rho1 = 10;pm_L1dL2.rho2 = 1; pm_L1dL2.beta = 10;
            pm_L1.rho = 10; pm_L1.lambda = 0.05; pm_L1.beta = 10; 
    end

I = reshape(xtrue,256,256);

% L1/L2-grad
    timestart = tic;
    [u_l1dl2,output_l1dl2] = mCTrecon_L1dL2_const(A, btrue, pm_L1dL2);
    timeout_l1dl2 = toc(timestart);
    fprintf('Relative error for L1/L2-grad: %3.3e\n',...
       output_l1dl2.err(end));
    toc
    output_l1dl2.u_orig=[];
    output_l1dl2.u=[];

