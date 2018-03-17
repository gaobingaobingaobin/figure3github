function [shadowedricianRV] = ShadowGenerator(m,n,b1,m1,omega1)
% m：发射天线数目；
% n：接收天线数目；
% b1，m1，omega1的设置参考文献《(2003)A new simple model for land mobile satellite
% channels first- and second-order statistics》中Table III
% 函数说明：
% 1、本函数对应阴影莱斯信道；
% 2、本函数的输出是信道的增益，在求解接收信噪比的时候所使用的功率增益是本函数输出值与其共轭转置的乘积。

RayleighRV = sqrt(b1)*(randn(m,n)+j*randn(m,n)); % Rayleigh distribution

mu=m1*ones(1:length(n));
omega=omega1*ones(1:length(n));
NakagamiRV=sqrt(gamrnd(mu,omega./mu,m,n));  % Nakagami distribution

zeta=0;
shadowedricianRV=RayleighRV+NakagamiRV*exp(j*zeta); % ShadowedRician distribution
end