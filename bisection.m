function c=bisection(m,g_sp,rho,gamma,eta,epsilon)
% 二分法，通过函数值反求Marcum Q function的自变量p_s
if nargin<6
epsilon=1e-5;
end
a=10^-5;
b=10^5;
% fa=f(a); %计算f(a)的值
fa=marcumq_var(m,a,g_sp,rho,gamma)-eta;
% fb=f(b); %计算f(b)的值
fb=marcumq_var(m,b,g_sp,rho,gamma)-eta;
c=(a+b)/2; %计算区间中点
% fc=f(c); %计算区间中点f(c)
fc=marcumq_var(m,c,g_sp,rho,gamma)-eta;
while abs(fc)>=epsilon; %判断f(c)是否为零点
if fa*fc>=0; %判断左侧区间是否有根
fa=fc;
a=c;
elseif fc*fb>=0
b=c;
end
c=(a+b)/2;
% fc=f(c);
fc=marcumq_var(m,c,g_sp,rho,gamma)-eta;
fa=marcumq_var(m,a,g_sp,rho,gamma)-eta;
fb=marcumq_var(m,b,g_sp,rho,gamma)-eta;
end
% fc
end
function y=marcumq_var(m,p_s,g_sp,rho,gamma,eta_2,eta_1)
if nargin<=5
    eta_1=1;
    eta_2=1;
end
% y=marcumq(  g_sp * sqrt (2 * rho^2 /  ( (1-rho^2) * eta_1 ) ), sqrt( 2*gamma / (p_s * eta_2 * (1-rho^2)  )  ),m  );
%% from the paper "generation of bivariate raleigh nd nakagami-m fading envelopes"
y=marcumq(  g_sp * sqrt (2 * rho^2 /  ( (1-rho^2) * eta_1 ) ), gamma / p_s * sqrt( 2 / (eta_2 * (1-rho^2)  )  ),m  );
end


