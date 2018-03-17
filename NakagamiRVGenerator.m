function [nakagamiRV] = NakagamiRVGenerator(m,n,m1,omega1)
% m：发射天线数目；
% n：接收天线数目；
% omega1：平均功率，值越大，表示信道条件越好
% m1：Nakagami信道的衰落参数，值越大表示信道条件越好

% 函数说明：
% 1、本函数对应Nakagami信道；
% 2、本函数的输出是信道的增益，在求解接收信噪比的时候所使用的功率增益是本函数输出值与其共轭转置的乘积。

mu=m1*ones(1:length(n));
omega=omega1*ones(1:length(n));
nakagamiRV=sqrt(gamrnd(mu,omega./mu,m,n));  % Nakagami distribution

end