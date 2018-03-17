% ������·����

function [h g] =LinkBudget() % ������·�������---g ������·�������---h

M = 5; % �����û���Ŀ
c = 3e8; % ����
% f = 28e9; % �ź�Ƶ�� %k1 Ƶ��
f = 2e9; % �ź�Ƶ��
Gt_max = 42.1; % FSS�����������棬��λ��dB
Gs_max = 52; % ���������������棬��λ��dB

% ������·�������---g
% h = zeros(1,M);

d = [120 78 53 89 67]*1e3; % ���ǵ���վ��FS��վ�ľ��룬��λ����

Ls = (c./(4.*pi.*d.*f)).^2; % ���ɿռ䴫�����
Ls_dB = 10*log10(Ls);

Fou = [79 138 68 116 33]; % ��λ������
Theta = [49 118 122 29 21]; % ��λ������

% Theta = [20 20 20 20 20]; % ��λ������
Gt = zeros(1,M);
Gr = zeros(1,M);
for i = 1:M
   if Fou(i) <= 1
       Gt(i) = Gt_max;
   elseif Fou(i) <= 48
       Gt(i) = 32-25*log10(Fou(i));
   else
       Gt(i) = -10;
   end
end

for i = 1:M
   if Theta(i) <= 20
       Gr(i) = 29-25*log10(Theta(i));
   elseif Theta(i) <= 26.3
       Gr(i) = -3.5;
   elseif Theta(i) <= 48
       Gr(i) = 32-25*log10(Theta(i));
   else 
       Gr(i) = -10;
   end
end

GI = Gt + Gr + Ls_dB; % ��λ��dB
g = 10.^(GI./10);

% ������·�������---h
L_dB = -212.5; % ���ɿռ䴫�����
% L_dB = -140; % ���ɿռ䴫�����
Lr = -[8.73,6.51,4.03,3.95,9.43]; % ��˥����λ��dB

fai = 0.1;
fai_3db = 0.4;
u1 = 2.07123 * sin(fai*pi/180)/sin(fai_3db*pi/180);
gs = ( besselj(1,u1)/(2*u1) + 36*besselj(3,u1)/u1^3 )^2; % sat beam gain
G_sat = Gs_max * gs;

GS = Gt_max + G_sat + L_dB + Lr;
h = 10.^(GS./10);
