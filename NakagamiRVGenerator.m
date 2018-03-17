function [nakagamiRV] = NakagamiRVGenerator(m,n,m1,omega1)
% m������������Ŀ��
% n������������Ŀ��
% omega1��ƽ�����ʣ�ֵԽ�󣬱�ʾ�ŵ�����Խ��
% m1��Nakagami�ŵ���˥�������ֵԽ���ʾ�ŵ�����Խ��

% ����˵����
% 1����������ӦNakagami�ŵ���
% 2����������������ŵ������棬������������ȵ�ʱ����ʹ�õĹ��������Ǳ��������ֵ���乲��ת�õĳ˻���

mu=m1*ones(1:length(n));
omega=omega1*ones(1:length(n));
nakagamiRV=sqrt(gamrnd(mu,omega./mu,m,n));  % Nakagami distribution

end