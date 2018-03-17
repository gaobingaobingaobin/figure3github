function [shadowedricianRV] = ShadowGenerator(m,n,b1,m1,omega1)
% m������������Ŀ��
% n������������Ŀ��
% b1��m1��omega1�����òο����ס�(2003)A new simple model for land mobile satellite
% channels first- and second-order statistics����Table III
% ����˵����
% 1����������Ӧ��Ӱ��˹�ŵ���
% 2����������������ŵ������棬������������ȵ�ʱ����ʹ�õĹ��������Ǳ��������ֵ���乲��ת�õĳ˻���

RayleighRV = sqrt(b1)*(randn(m,n)+j*randn(m,n)); % Rayleigh distribution

mu=m1*ones(1:length(n));
omega=omega1*ones(1:length(n));
NakagamiRV=sqrt(gamrnd(mu,omega./mu,m,n));  % Nakagami distribution

zeta=0;
shadowedricianRV=RayleighRV+NakagamiRV*exp(j*zeta); % ShadowedRician distribution
end