function [u,output] = mCTrecon_L1dL2_const(A, f, pm)
%=============================================================
% CT reconstruction via L1/l2 on the gradient
%
% Solves
%           min  norm(\nabla x(:),1)/norm(\nabla x(:),2)
%           s.t. Ax = b, p<=x<=q 
%
% Author: Chao Wang
% Date: July 19. 2019
% solve it by splitting scheme
%=============================================================
output = pm;
rows = pm.rows; cols = pm.cols; 
%% default setting and set up parameter from pm
lambda = 30; rho1=1; rho2 =1; beta = 1; maxit = 100;Imaxit = 1;
u_orig = zeros(rows,cols); 
u0 = zeros(rows,cols); 
tol = 1e-5;
StopCri = 1; box = 1;

if isfield(pm,'rho1'); rho1 = pm.rho1; end
if isfield(pm,'rho2'); rho2 = pm.rho2; end
if isfield(pm,'beta'); beta = pm.beta; end
if isfield(pm,'lambda'); lambda = pm.lambda; end
if isfield(pm,'maxit'); maxit = pm.maxit; end
if isfield(pm,'Imaxit'); Imaxit = pm.Imaxit; end
if isfield(pm,'u_orig');u_orig = pm.u_orig; end
if isfield(pm,'u0');u0 = pm.u0; end
if isfield(pm,'tol'); tol = pm.tol; end
if isfield(pm,'StopCri'); StopCri = pm.StopCri; end
if isfield(pm,'box'); box = pm.box; end
%% Reserve memory for the auxillary variables
u = u0; 
dx = zeros(rows,cols);
dy = zeros(rows,cols);
hx = zeros(rows,cols);
hy = zeros(rows,cols);
bx = zeros(rows,cols);
by = zeros(rows,cols);
cx = zeros(rows,cols);
cy = zeros(rows,cols);
w  = zeros(size(f));
h = bx;
%% parameter setting of conjugate gradient for solving the subproblem.
cgpm = pm; cgpm.gamma = rho1 + rho2;

%% output list
list_re = zeros(maxit,1);list_e = list_re;
list_u = zeros(rows,cols, maxit);
list_o = list_re;
list_c = list_re;
list_cpu = list_re;
tstart = tic;
h_wait = waitbar(0, 'Running iterations for L1/L2, please wait ...');
step = 0; 
list_fo = zeros(maxit*Imaxit,1);
list_fe = zeros(maxit*Imaxit,1);
list_fc = zeros(maxit*Imaxit,1);
%% Algorithm ADMM
for j = 1: maxit
    waitbar(j/maxit, h_wait)
    uold = u; v = u;
    for i = 1: Imaxit
        % u-update
        step = step + 1; 
        uold_inner = u;
        rhs = reshape(lambda * A' * (f + w), rows,cols) +rho1*Dxt(dx-bx)...
            + rho1*Dyt(dy-by)+rho2*Dxt(hx-cx)+rho2*Dyt(hy-cy);
        if box == 1
            rhs = rhs + beta*(v-h);
            [tmp, ~] = conjgrad_b(A, rhs(:), u(:), cgpm);
            u = reshape(tmp, rows, cols);
            v = u+h;
            v(v<0)=0; v(v>1) =1;
            h  = h + u - v;
        else
            [tmp, ~] = conjgrad(A, rhs(:), u(:), cgpm);
            u = reshape(tmp, rows, cols);
        end
        list_cpu(j) = toc(tstart);
        % w-update
        w = w + f-A*u(:);
        % dx,dy-update (gradient of u)
        Dxu = Dx(u);
        Dyu = Dy(u);
        hnorm = sqrt(norm(hx(:))^2+norm(hy(:))^2);
        dx = shrink(Dxu+bx, 1/(rho1*hnorm));
        dy = shrink(Dyu+by, 1/(rho1*hnorm));
        bx = bx+(Dxu-dx);
        by = by+(Dyu-dy);
%         list_fo(step) =  (norm(Dxu(:),1)+norm(Dyu(:),1))/sqrt(norm(Dxu,'fro').^2+norm(Dyu,'fro').^2);
%         list_fe(step) =  norm(u-u_orig,'fro')/norm(u_orig,'fro');
%         list_fc(step) = norm(u-u_orig,'fro')/sqrt(numel(u));
        if norm(uold_inner-u,'fro')/norm(u,'fro')<1e-5
            break
        end
        
        
    end
    % hx,hy-update
    d1 = Dxu + cx;
    d2 = Dyu + cy;
    etha = sqrt(norm(d1(:))^2+norm(d2(:))^2); 
    c = norm(dx(:),1)+norm(dy(:),1);
    [hx, hy,~] = mupdate_h(c,etha,rho2,d1,d2);

    % cx,cy-update (updating d)
    
    cx = cx+(Dxu-hx);
    cy = cy+(Dyu-hy);
    
    % eveluation
    list_re(j) = norm(uold-u,'fro')/norm(u,'fro');
%     list_e(j)=  norm(u-u_orig,'fro')/norm(u_orig,'fro');
%     list_o(j) = (norm(Dxu(:),1)+norm(Dyu(:),1))/sqrt(norm(Dxu,'fro').^2+norm(Dyu,'fro').^2);
    list_c(j) = norm(u-u_orig,'fro')/sqrt(numel(u));
    list_u(:,:,j) = u;
     % stopping condition
    if StopCri ==1
        relerr = list_re(j);
        if relerr <tol && j > 50
            break
        end
    end

%     if fix(j/10) == j/10
%         imshow(u);
%         title(['iteration:' num2str(j) ' ;RE:' num2str(list_e(j))])
%         pause(0.2)
%     end
end
list_re(j+1:end)=[]; list_e(j+1:end) = [];list_o(j+1:end) = [];
list_c(j+1:end) = [];list_u(:,:,j+2:end) = [];

output.relerr = list_re;
output.err = list_e;
output.ferr = list_fe;
output.u = list_u;
output.obj = list_o;
output.fobj = list_fo;
output.rmse = list_c;
output.frmse = list_fc;
output.cpu = list_cpu;
% Dxu_gt = Dx(u_orig); Dyu_gt = Dy(u_orig);
% output.obj_gt = (norm(Dxu_gt(:),1)+norm(Dyu_gt(:),1))/sqrt(norm(Dxu_gt,'fro').^2+norm(Dyu_gt,'fro').^2);
close(h_wait)
return;

%% gradient operators
function d = Dx(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
d(:,1) = u(:,1)-u(:,cols);
return;

function d = Dxt(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
d(:,cols) = u(:,cols)-u(:,1);
return;

function d = Dy(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
d(1,:) = u(1,:)-u(rows,:);
return;

function d = Dyt(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
d(rows,:) = u(rows,:)-u(1,:);
return;

function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
return;


function [hx, hy,tau] = mupdate_h(c,etha,rho,d1,d2)

if etha == 0 
   hx = (c/rho)^(1/3)*ones(size(d1))/sqrt(numel(d1)*2);
   hy = (c/rho)^(1/3)*ones(size(d2))/sqrt(numel(d2)*2);
else
    a = 27*c/(rho*(etha^3)) + 2;
    C = ((a + (a^2 - 4)^0.5)/2)^(1/3);
    tau = (1 + C + 1/C)/3;
    hx = tau*d1;
    hy = tau*d2;
end
return;

function [x,rsnew] = conjgrad(A,b,x,pm)
    r=b-Ax(A,x,pm);
    p=r;
    rsold=r'*r;
 
    for i=1:sqrt(pm.rows)
        Ap=Ax(A,p,pm);
        alpha=rsold/(p'*Ap);
        x=x+alpha*p;
        r=r-alpha*Ap;
        rsnew=r'*r;
        if sqrt(rsnew)<1e-2
              break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
return;
    
    
function y = Ax(P,x, pm)

g = fspecial('laplacian',0);
y = imfilter(resh ape(x,pm.rows,pm.cols),g, 'circular');
y =  pm.lambda*P'*(P*x(:))-pm.gamma*y(:);
return;

function [x,rsnew] = conjgrad_b(A,b,x,pm)
    r=b-Ax_b(A,x,pm);
    p=r;
    rsold=r'*r;
 
    for i=1:sqrt(pm.rows)
        Ap=Ax_b(A,p,pm);
        alpha=rsold/(p'*Ap);
        x=x+alpha*p;
        r=r-alpha*Ap;
        rsnew=r'*r;
        if sqrt(rsnew)<1e-2
              break;
        end
        p=r+rsnew/rsold*p;
        rsold=rsnew;
    end
return;
    
    
function y = Ax_b(P,x, pm)

g = fspecial('laplacian',0);
y = imfilter(reshape(x,pm.rows,pm.cols),g, 'circular');
y =  pm.lambda*P'*(P*x(:))-pm.gamma*y(:)+pm.beta*x(:);
return;