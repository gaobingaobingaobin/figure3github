% 《Optimal power allocation for Bi-directional cognitive radio networks using alternating optimzation》
%本程序的假设是2个主接收用户PR，5个次级用户ST
% ii 是 Pout的迭代指标
clc;
clear;
close all;
B=50; %bandiwith
% 系统参数设置
M = 1e2;% 衰落信道状态数
[Psi, Phi] =LinkBudget(); %Psi 卫星次级链路的增益 ； Phi 干扰链路增益
% NN = 1e3;% 迭代次数
sigma1 = 1e-13;% h1噪声功率% to balance Psi into the same exp order， e.g., Psi=1.1e-13,sigma1=1e-13, then Psi/sigma1=1.1
sigma2 = 1e-13;% h2噪声功率
sigma3 = 1e-13;% h3噪声功率
sigma4 = 1e-13;% h4噪声功率
sigma5 = 1e-13;% h5噪声功率
% beta1 = 1e-2;
% beta2 = 1e-2;
% beta01=1e-2;
% beta
lambda0 = 0;

% 精度要求
Epsilon1 = 1e-3;
Epsilon2 = 1e-3;
Epsilon3 = 1e-3;
Epsilon4 = 1e-3;
Epsilon5 = 1e-3;

b_SL=[0.158 0.063 0.126];% 分别为infrequent light shadow； frequent heavy shadowing； average shadowing
m_SL=[19.4 0.739 10.1]; % 注意原文献的0.739，石盛超的改为2
Omega_SL=[1.29 8.97e-4 0.835];

% Rayleigh channel
g11 = zeros(1,M);
g12 = zeros(1,M);
g13 = zeros(1,M);
g14 = zeros(1,M);
g15 = zeros(1,M);
g21 = zeros(1,M);
g22 = zeros(1,M);
g23 = zeros(1,M);
g24 = zeros(1,M);
g25 = zeros(1,M);
h1 = zeros(1,M);
h2 = zeros(1,M);
h3 = zeros(1,M);
h4 = zeros(1,M);
h5 = zeros(1,M);
P11out=zeros(1,M);
P12out=zeros(1,M);
P13out=zeros(1,M);
P14out=zeros(1,M);
P15out=zeros(1,M);
P21out=zeros(1,M);
P22out=zeros(1,M);
P23out=zeros(1,M);
P24out=zeros(1,M);
P25out=zeros(1,M);
P1out=zeros(1,M);%  2个PU面对1th SU最小的功率上限 from outage constraint
P2out=zeros(1,M);%  2个PU面对2th SU最小的功率上限 from outage constraint
P3out=zeros(1,M);
P4out=zeros(1,M);
P5out=zeros(1,M);
% etaRange=0.1:0.1:1;% outage constraint from eq. P(g_mi * g_i>gamma_mi)<eta_mi
% etaRange =[0.4,0.6,0.9];
etaRange =[0.6];
gamma_mi=0.1; % from eq. P(g_mi * g_i>gamma_mi)<eta_mi
mlink = 1;
Omega_g11 = 2;% 次级用户1针对主用户1的信道增益gain
Omega_g12 = 1;% 次级用户2针对主用户1的信道增益gain
Omega_g13 = 1;
Omega_g14 = 1;
Omega_g15 = 1;
Omega_g21 = 2;
Omega_g22 = 1;
Omega_g23 = 1;% 次级用户3针对主用户2的信道增益gain
Omega_g24 = 1;% 次级用户4针对主用户2的信道增益gain
Omega_g25 = 1;% 次级用户5针对主用户2的信道增益gain

Omega_h1 = 2;
Omega_h2 = 1.5;
Omega_h3 = 1;
Omega_h4 = 3;
Omega_h5 = 2.5;

for i = 1:M
    a11 = NakagamiRVGenerator(1,1,mlink,Omega_g11);
    g11(i) = a11*a11';
    a12 = NakagamiRVGenerator(1,1,mlink,Omega_g12);
    g12(i) = a12*a12';
    a13 = NakagamiRVGenerator(1,1,mlink,Omega_g13);
    g13(i) = a13*a13';
    a14 = NakagamiRVGenerator(1,1,mlink,Omega_g14);
    g14(i) = a14*a14';
    a15 = NakagamiRVGenerator(1,1,mlink,Omega_g15);
    g15(i) = a15*a15';
    a21 = NakagamiRVGenerator(1,1,mlink,Omega_g21);
    g21(i) = a21*a21';
    a22 = NakagamiRVGenerator(1,1,mlink,Omega_g22);
    g22(i) = a22*a22';
    a23 = NakagamiRVGenerator(1,1,mlink,Omega_g23);
    g23(i) = a23*a23';
    a24 = NakagamiRVGenerator(1,1,mlink,Omega_g24);
    g24(i) = a24*a24';
    a25 = NakagamiRVGenerator(1,1,mlink,Omega_g25);
    g25(i) = a25*a25';
    
    
    
    ii=1;% infrequent light shadow
     
    aa1 = ShadowGenerator(1,1,b_SL(ii),m_SL(ii),Omega_SL(ii));% 发射天线1 接收天线1
    h1(i) = aa1*aa1';
    aa11 = ShadowGenerator(1,1,b_SL(ii),m_SL(ii),Omega_SL(ii));% 发射天线1 接收天线1
    h2(i) = aa11*aa11';
    aa2 = ShadowGenerator(1,1,b_SL(ii),m_SL(ii),Omega_SL(ii));% 发射天线1 接收天线1
    h3(i) = aa2*aa2';
    aa3 = ShadowGenerator(1,1,b_SL(ii),m_SL(ii),Omega_SL(ii));% 发射天线1 接收天线1
    h4(i) = aa3*aa3';
    aa4 = ShadowGenerator(1,1,b_SL(ii),m_SL(ii),Omega_SL(ii));% 发射天线1 接收天线1
    h5(i) = aa4*aa4';
end
jj=1;
rho=0.9;% the correlation coefficient: oudated CSI中很重要的一个参数
% I_dB = -20:2:10;% dB值
I_dB = 10;% dB值 : 事实上IdB=-160，但为了实验数据实现漂亮，1e-16放在gVec1和gVec2的分母上了
% I_dB =-160;% to balance the order of Phi, i.e., the interference link gain
Irange = 10.^(I_dB/10);% 转化成真值
Len = length(Irange);
ECsumgao=zeros(3,Len);
% for ii=1:length(etaRange)
    %% outage constraint transformed to peak constraint: eq. P(g_mi * g_i>gamma_mi)<eta_mi =>g_i<P_mi^out
%     eta_mi=etaRange(ii);
eta_mi=etaRange;
    for i = 1 : M
        P11out(i)=bisection(mlink,g11(i),rho,gamma_mi,eta_mi);
        P12out(i)=bisection(mlink,g12(i),rho,gamma_mi,eta_mi);
        P13out(i)=bisection(mlink,g13(i),rho,gamma_mi,eta_mi);
        P14out(i)=bisection(mlink,g14(i),rho,gamma_mi,eta_mi);
        P15out(i)=bisection(mlink,g15(i),rho,gamma_mi,eta_mi);
        P21out(i)=bisection(mlink,g21(i),rho,gamma_mi,eta_mi);
        P22out(i)=bisection(mlink,g22(i),rho,gamma_mi,eta_mi);
        P23out(i)=bisection(mlink,g23(i),rho,gamma_mi,eta_mi);
        P24out(i)=bisection(mlink,g24(i),rho,gamma_mi,eta_mi);
        P25out(i)=bisection(mlink,g25(i),rho,gamma_mi,eta_mi);
        P1out(i)=min(P11out(i),P21out(i));
        P2out(i)=min(P12out(i),P22out(i));
        P3out(i)=min(P13out(i),P23out(i));
        P4out(i)=min(P14out(i),P24out(i));
        P5out(i)=min(P15out(i),P25out(i));
    end
    h1 = h1.*Psi(1)./sigma1;
    h2 = h2.*Psi(2)./sigma2;
    h3 = h3.*Psi(3)./sigma3;
    h4 = h4.*Psi(4)./sigma4;
    h5 = h5.*Psi(5)./sigma5;
    hVec=[h1;h2;h3;h4;h5];
    gVec1=[g11.*Phi(1);g12.*Phi(2);g13.*Phi(3);g14.*Phi(4);g15.*Phi(5)].*1e16;
    gVec2=[g21.*Phi(1);g22.*Phi(1);g23.*Phi(1);g24.*Phi(1);g25.*Phi(1)].*1e16;
    
    save hVec;
    save gVec1;
    save gVec2;
    save PoutVec;
    NN = 1e3;
    %sensing 部分的假定
%     P1H0=0.4;%用户1 inactive 概率
%     P1H1=0.6;%用户1 active 概率
%     P2H0=0.3;%用户2 inactive 概率
%     P2H1=0.7;%用户2 active 概率
    %without sensing 部分的假定
    P1H0=0;%用户1 inactive 概率
    P1H1=1;%用户1 active 概率
    P2H0=0;%用户2 inactive 概率
    P2H1=1;%用户2 active 概率
    PHvec=[P1H0,P1H1,P2H0,P2H1];
    % 干扰门限
    % I_dB = -20:2:20;% dB值 10之后貌似有些跳跃
%     I_dB = -20:2:10;% dB值
    %     Irange = 10.^(I_dB/10);% 转化成真值
    %     Len = length(Irange);
    %     ECsumgao=zeros(3,Len);
    
    EC1_00 = zeros(1,NN);
    EC2_00 = zeros(1,NN);
    EC3_00 = zeros(1,NN);
    EC4_00 = zeros(1,NN);
    EC5_00 = zeros(1,NN);
    ECsum_00=zeros(1,NN);
    EC1_01 = zeros(1,NN);
    EC2_01 = zeros(1,NN);
    EC3_01 = zeros(1,NN);
    EC4_01 = zeros(1,NN);
    EC5_01 = zeros(1,NN);
    ECsum_01=zeros(1,NN);
    EC1_10 = zeros(1,NN);
    EC2_10 = zeros(1,NN);
    EC3_10 = zeros(1,NN);
    EC4_10 = zeros(1,NN);
    EC5_10 = zeros(1,NN);
    ECsum_10=zeros(1,NN);
    EC1_11 = zeros(1,NN);
    EC2_11 = zeros(1,NN);
    EC3_11 = zeros(1,NN);
    EC4_11 = zeros(1,NN);
    EC5_11 = zeros(1,NN);
    ECsum_11=zeros(1,NN);
    ECsum=zeros(1,NN);
    ECsumMat=zeros(3,NN);
%     for i = 1:Len
%         I1 = Irange(i);% interference constraint from sum g_1i * q_i< I1
        I1 = Irange;
        I2=I1+0.01;% interference constraint from sum g_2i * q_i< I2
        P1av=1+5;
        P2av=1+6;%次级用户2平均功率上限
        P3av=1+6;%次级用户3平均功率上限
        P4av=1+6;%次级用户4平均功率上限
        P5av=1+6;%次级用户5平均功率上限
        PavVec=[P1av,P2av,P3av,P4av,P5av];
        phi1_00 = zeros(1,M);
        phi1_01 = zeros(1,M);
        phi1_10 = zeros(1,M);
        phi1_11 = zeros(1,M);
        
        phi2_01 = zeros(1,M);
        phi2_10 = zeros(1,M);
        phi2_11 = zeros(1,M);
        phi2_00 = zeros(1,M);
        phi3_01 = zeros(1,M);
        phi3_10 = zeros(1,M);
        phi3_11 = zeros(1,M);
        phi3_00 = zeros(1,M);
        phi4_01 = zeros(1,M);
        phi4_10 = zeros(1,M);
        phi4_11 = zeros(1,M);
        phi4_00 = zeros(1,M);
        phi5_01 = zeros(1,M); % 5 对应次级用户j，01代表主用户1 inactive ，主用户2 active
        phi5_10 = zeros(1,M);
        phi5_11 = zeros(1,M);
        phi5_00 = zeros(1,M);
        varphi1_01 = zeros(1,M); % 1 对应次级用户j，01代表主用户1 inactive ，主用户2 active
        varphi1_10 = zeros(1,M);
        varphi1_11 = zeros(1,M);
        varphi2_01 = zeros(1,M); % 2 对应次级用户j，01代表主用户1 inactive ，主用户2 active
        varphi2_10 = zeros(1,M);
        varphi2_11 = zeros(1,M);
        varphi3_01 = zeros(1,M); % 3 对应次级用户3，01代表主用户1 inactive ，主用户2 active
        varphi3_10 = zeros(1,M);
        varphi3_11 = zeros(1,M);
        varphi4_01 = zeros(1,M); % 4 对应次级用户4，01代表主用户1 inactive ，主用户2 active
        varphi4_10 = zeros(1,M);
        varphi4_11 = zeros(1,M);
        varphi5_01 = zeros(1,M); % 5 对应次级用户5，01代表主用户1 inactive ，主用户2 active
        varphi5_10 = zeros(1,M);
        varphi5_11 = zeros(1,M);
        
        p1_00 = zeros(1,M); %Prob1(H_0)和Prob2(H_0)假定，i.e.,主用户1 inactive 且主用户2 inactive 次级用户1的发射功率
        p2_00 = zeros(1,M); %Prob1(H_0)和Prob2(H_0)假定，i.e.,主用户1 inactive 且主用户2 inactive 次级用户2的发射功率
        p3_00 = zeros(1,M); %Prob1(H_0)和Prob2(H_0)假定，i.e.,主用户1 inactive 且主用户2 inactive 次级用户3的发射功率
        p4_00= zeros(1,M);  %Prob1(H_0)和Prob2(H_0)假定，i.e.,主用户1 inactive 且主用户2 inactive 次级用户4的发射功率
        p5_00 = zeros(1,M); %Prob1(H_0)和Prob2(H_0)假定，i.e.,主用户1 inactive 且主用户2 inactive 次级用户5的发射功率
        p1_01 = zeros(1,M); %Prob1(H_0)和Prob2(H_1)假定，i.e.,主用户1 inactive 且主用户2 active 次级用户1的发射功率
        p2_01 = zeros(1,M); %Prob1(H_0)和Prob2(H_1)假定，i.e.,主用户1 inactive 且主用户2 active 次级用户2的发射功率
        p3_01 = zeros(1,M); %Prob1(H_0)和Prob2(H_1)假定，i.e.,主用户1 inactive 且主用户2 active 次级用户3的发射功率
        p4_01 = zeros(1,M); %Prob1(H_0)和Prob2(H_1)假定，i.e.,主用户1 inactive 且主用户2 active 次级用户4的发射功率
        p5_01 = zeros(1,M); %Prob1(H_0)和Prob2(H_1)假定，i.e.,主用户1 inactive 且主用户2 active 次级用户5的发射功率
        p1_10 = zeros(1,M); %Prob1(H_1)和Prob2(H_0)假定，i.e.,主用户1 active 且主用户2 inactive 次级用户1的发射功率
        p2_10 = zeros(1,M); %Prob1(H_1)和Prob2(H_0)假定，i.e.,主用户1 active 且主用户2 inactive 次级用户2的发射功率
        p3_10 = zeros(1,M); %Prob1(H_1)和Prob2(H_0)假定，i.e.,主用户1 active 且主用户2 inactive 次级用户3的发射功率
        p4_10 = zeros(1,M); %Prob1(H_1)和Prob2(H_0)假定，i.e.,主用户1 active 且主用户2 inactive 次级用户4的发射功率
        p5_10 = zeros(1,M); %Prob1(H_1)和Prob2(H_0)假定，i.e.,主用户1 active 且主用户2 inactive 次级用户5的发射功率
        p1_11 = zeros(1,M); %Prob1(H_1)和Prob2(H_1)假定，i.e.,主用户1 active 且主用户2 active 次级用户1的发射功率
        p2_11 = zeros(1,M); %Prob1(H_1)和Prob2(H_1)假定，i.e.,主用户1 active 且主用户2 active 次级用户2的发射功率
        p3_11 = zeros(1,M); %Prob1(H_1)和Prob2(H_1)假定，i.e.,主用户1 active 且主用户2 active 次级用户3的发射功率
        p4_11 = zeros(1,M); %Prob1(H_1)和Prob2(H_1)假定，i.e.,主用户1 active 且主用户2 active 次级用户4的发射功率
        p5_11 = zeros(1,M); %Prob1(H_1)和Prob2(H_1)假定，i.e.,主用户1 active 且主用户2 active 次级用户5的发射功率
        Delta1 = 1;
        Delta2 = 1;
        Delta3 = 1;
        Delta4 = 1;
        Delta5 = 1;
        k = 1;
        lambda1 = zeros(1,M);%针对次级用户1的平均功率门限P_1^av约束langrange乘子 for sum_{i_1,i_2 \in{0,1}}P(H_{i_1})P(H_{i_2})<P_j^av
        lambda2 = zeros(1,M);%针对次级用户2的平均功率门限P_1^av约束langrange乘子
        lambda3 = zeros(1,M);%针对次级用户3的平均功率门限P_1^av约束langrange乘子
        lambda4 = zeros(1,M);%针对次级用户4的平均功率门限P_1^av约束langrange乘子
        lambda5 = zeros(1,M);%针对次级用户5的平均功率门限P_1^av约束langrange乘子
        lambdaVec=[lambda1;lambda2;lambda3;lambda4;lambda5];
        for iii=0:2  % variation of beta =[1e-2,1e,1e2]
            aa=iii*2;
            aaa=10^aa;
        beta1=1e-2*aaa;beta2=1e-2*aaa; beta3=1e-2*aaa;beta4=1e-2*aaa; beta5=1e-2*aaa;
        betaVec=[beta1,beta2,beta3,beta4,beta5];
        beta10=1e-2*aaa;
        
        beta01=1e-2*aaa;
        
        beta111=1e-2*aaa;
        
        beta112=1e-2*aaa;
        
        lambda10= zeros(1,M);%针对主用户1为active主用户2为inactive的针对主用户1门限为I1的约束langrange乘子次级用户1
        
        lambda01= zeros(1,M);%针对主用户1为inactive主用户2为active的针对主用户1门限为I2的约束langrange乘子针对所有次级用户
        
        lambda111 = zeros(1,M);%针对主用户1为active主用户2为active的针对主用户2门限为I1的约束langrange乘子针对所有次级用户
        
        lambda112 = zeros(1,M);%针对主用户1为active主用户2为active的针对主用户2门限为I2的约束langrange乘子针对所有次级用户
        
        
        for nn = 1:NN
            p1_00buf = p1_00;
            p2_00buf = p2_00;
            p3_00buf = p3_00;
            p4_00buf = p4_00;
            p5_00buf = p5_00;
            p1_01buf = p1_01;
            p2_01buf = p2_01;
            p3_01buf = p3_01;
            p4_01buf = p4_01;
            p5_01buf = p5_01;
            p1_10buf = p1_10;
            p2_10buf = p2_10;
            p3_10buf = p3_10;
            p4_10buf = p4_10;
            p5_10buf = p5_10;
            p1_11buf = p1_11;
            p2_11buf = p2_11;
            p3_11buf = p3_11;
            p4_11buf = p4_11;
            p5_11buf = p5_11;
            p_00Vec=[p1_00;p2_00;p3_00;p4_00;p5_00];
            p_01Vec=[p1_01;p2_01;p3_01;p4_01;p5_01];
            p_10Vec=[p1_10;p2_10;p3_10;p4_10;p5_10];
            p_11Vec=[p1_11;p2_11;p3_11;p4_11;p5_11];
            
            %% pj00 子问题
            phi1_00=beta1.*P1H0.*P2H0.*(P1H0*P2H1.* p1_01buf + P1H1*P2H0.*p1_10buf + P1H1*P2H1.*p1_11buf-P1av)-lambda1.* P1H0.*P2H0;
            phi2_00=beta2.*P1H0.*P2H0.*(P1H0*P2H1.* p2_01buf + P1H1*P2H0.*p2_10buf + P1H1*P2H1.*p2_11buf-P1av)-lambda2.* P1H0.*P2H0;
            phi3_00=beta3.*P1H0.*P2H0.*(P1H0*P2H1.* p3_01buf + P1H1*P2H0.*p3_10buf + P1H1*P2H1.*p3_11buf-P1av)-lambda3.* P1H0.*P2H0;
            phi4_00=beta4.*P1H0.*P2H0.*(P1H0*P2H1.* p4_01buf + P1H1*P2H0.*p4_10buf + P1H1*P2H1.*p4_11buf-P1av)-lambda4.* P1H0.*P2H0;
            phi5_00=beta5.*P1H0.*P2H0.*(P1H0*P2H1.* p5_01buf + P1H1*P2H0.*p5_10buf + P1H1*P2H1.*p5_11buf-P1av)-lambda5.* P1H0.*P2H0;
            a1_00=beta1.*(P1H0.* P2H0).^2.*h1;
            a2_00=beta2.*(P1H0.* P2H0).^2.*h2;
            a3_00=beta3.*(P1H0.* P2H0).^2.*h3;
            a4_00=beta4.*(P1H0.* P2H0).^2.*h4;
            a5_00=beta5.*(P1H0.* P2H0).^2.*h5;
            b1_00=h1.*phi1_00;b2_00=h2.*phi2_00;b3_00=h3.*phi3_00;b4_00=h4.*phi4_00;b5_00=h1.*phi5_00;
            c1_00=sigma1.*phi1_00-h1.*P1H0.*P2H0;
            c2_00=sigma2.*phi2_00-h2.*P1H0.*P2H0;
            c3_00=sigma3.*phi3_00-h3.*P1H0.*P2H0;
            c4_00=sigma4.*phi4_00-h4.*P1H0.*P2H0;
            c5_00=sigma5.*phi5_00-h5.*P1H0.*P2H0;
            p1_00tilde=(-b1_00+sqrt(b1_00.^2-4.*a1_00.*c1_00)) ./(2.*a1_00);
            p2_00tilde=(-b2_00+sqrt(b2_00.^2-4.*a2_00.*c2_00)) ./(2.*a2_00);
            p3_00tilde=(-b3_00+sqrt(b3_00.^2-4.*a3_00.*c3_00)) ./(2.*a3_00);
            p4_00tilde=(-b4_00+sqrt(b4_00.^2-4.*a4_00.*c4_00)) ./(2.*a4_00);
            p5_00tilde=(-b5_00+sqrt(b5_00.^2-4.*a5_00.*c5_00)) ./(2.*a5_00);
            
            p1_00bar=(-b1_00-sqrt(b1_00.^2-4.*a1_00.*c1_00)) ./(2.*a1_00);
            p2_00bar=(-b2_00-sqrt(b2_00.^2-4.*a2_00.*c2_00)) ./(2.*a2_00);
            p3_00bar=(-b3_00-sqrt(b3_00.^2-4.*a3_00.*c3_00)) ./(2.*a3_00);
            p4_00bar=(-b4_00-sqrt(b4_00.^2-4.*a4_00.*c4_00)) ./(2.*a4_00);
            p5_00bar=(-b5_00-sqrt(b5_00.^2-4.*a5_00.*c5_00)) ./(2.*a5_00);
            
            p_00tildeVec=[p1_00tilde;p2_00tilde;p3_00tilde;p4_00tilde;p5_00tilde];
            p_00barVec=[p1_00bar;p2_00bar;p3_00bar;p4_00bar;p5_00bar];
            %因为bj肯定大于零，所以左边的根肯定小于0，局部最大值再x负轴上，那么L（pj_00）就不需要跟L（0）作比较
            %错错错，不一定bj肯定大于零，要比较，稍后补上
            %5 个 pi00依次迭代
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi00(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,1); %1 :ST
                Lpi= lagrangePi00(p_00tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,1); %1 :ST
                if b1_00(channelNum)^2-4*a1_00(channelNum)*c1_00(channelNum) < 0
                    p1_00(channelNum) = 0;
                elseif p_00tildeVec(1,channelNum) >= 0&& Lpi<=Lpi0
                    p1_00(channelNum) = min(p_00tildeVec(1,channelNum),P1out(channelNum));
                elseif p_00barVec(1,channelNum) >= 0 && Lpi<=Lpi0
                    p1_00(channelNum) =  min(p_00tildeVec(1,channelNum),P1out(channelNum));
                else
                    p1_00(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi00(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,2); %1 :ST
                Lpi= lagrangePi00(p_00tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,2); %1 :ST
                if b2_00(channelNum)^2-4*a2_00(channelNum)*c2_00(channelNum) < 0
                    p2_00(channelNum) = 0;
                elseif p_00tildeVec(2,channelNum) >= 0&& Lpi<=Lpi0
                    p2_00(channelNum) =  min(p_00tildeVec(2,channelNum),P2out(channelNum));
                elseif p_00barVec(2,channelNum) >= 0 && Lpi<=Lpi0
                    p2_00(channelNum) =  min(p_00tildeVec(2,channelNum),P2out(channelNum));
                else
                    p2_00(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi00(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,3); %3 :ST
                Lpi= lagrangePi00(p_00tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,3); %3 :ST
                if b3_00(channelNum)^2-4*a3_00(channelNum)*c3_00(channelNum) < 0
                    p3_00(channelNum) = 0;
                elseif p_00tildeVec(3,channelNum) >= 0&& Lpi<=Lpi0
                    p3_00(channelNum) =  min(p_00tildeVec(3,channelNum),P3out(channelNum));
                elseif p_00barVec(3,channelNum) >= 0 && Lpi<=Lpi0
                    p3_00(channelNum) =  min(p_00tildeVec(3,channelNum),P3out(channelNum));
                else
                    p3_00(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi00(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,4); %4 :ST
                Lpi= lagrangePi00(p_00tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,4); %4 :ST
                if b4_00(channelNum)^2-4*a4_00(channelNum)*c4_00(channelNum) < 0
                    p4_00(channelNum) = 0;
                elseif p_00tildeVec(4,channelNum) >= 0&& Lpi<=Lpi0
                    p4_00(channelNum) =  min(p_00tildeVec(4,channelNum),P4out(channelNum));
                elseif p_00barVec(4,channelNum) >= 0 && Lpi<=Lpi0
                    p4_00(channelNum) =  min(p_00tildeVec(4,channelNum),P4out(channelNum));
                else
                    p4_00(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi00(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,5); %5 :ST
                Lpi= lagrangePi00(p_00tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,PavVec,channelNum,5); %5 :ST
                if b5_00(channelNum)^2-4*a5_00(channelNum)*c5_00(channelNum) < 0
                    p5_00(channelNum) = 0;
                elseif p_00tildeVec(5,channelNum) >= 0&& Lpi<=Lpi0
                    p5_00(channelNum) =  min(p_00tildeVec(5,channelNum),P5out(channelNum));
                elseif p_00barVec(5,channelNum) >= 0 && Lpi<=Lpi0
                    p5_00(channelNum) =  min(p_00tildeVec(5,channelNum),P5out(channelNum));
                else
                    p5_00(channelNum) = 0;
                end
            end
            
            
            
            
            
            %% pj01 子问题
            phi1_01=beta1.*P1H0.*P2H1.*(P1H0*P2H0.* p1_00buf + P1H1*P2H0.*p1_10buf + P1H1*P2H1.*p1_11buf-P1av)-lambda1.* P1H0.*P2H1;
            phi2_01=beta2.*P1H0.*P2H1.*(P1H0*P2H0.* p2_00buf + P1H1*P2H0.*p2_10buf + P1H1*P2H1.*p2_11buf-P2av)-lambda2.* P1H0.*P2H1;
            phi3_01=beta3.*P1H0.*P2H1.*(P1H0*P2H0.* p3_00buf + P1H1*P2H0.*p3_10buf + P1H1*P2H1.*p3_11buf-P3av)-lambda3.* P1H0.*P2H1;
            phi4_01=beta4.*P1H0.*P2H1.*(P1H0*P2H0.* p4_00buf + P1H1*P2H0.*p4_10buf + P1H1*P2H1.*p4_11buf-P4av)-lambda4.* P1H0.*P2H1;
            phi5_01=beta5.*P1H0.*P2H1.*(P1H0*P2H0.* p5_00buf + P1H1*P2H0.*p5_10buf + P1H1*P2H1.*p5_11buf-P5av)-lambda5.* P1H0.*P2H1;
            sum_g2ipi01=g21.*p1_01buf+g22.*p2_01buf+g23.*p3_01buf+g24.*p4_01buf+g25.*p5_01buf;
            varphi1_01=beta01.*g21.*(sum_g2ipi01-g21.*p1_01buf-I2)-lambda01.*g21;
            varphi2_01=beta01.*g22.*(sum_g2ipi01-g22.*p2_01buf-I2)-lambda01.*g22;
            varphi3_01=beta01.*g23.*(sum_g2ipi01-g23.*p3_01buf-I2)-lambda01.*g23;
            varphi4_01=beta01.*g24.*(sum_g2ipi01-g24.*p4_01buf-I2)-lambda01.*g24;
            varphi5_01=beta01.*g25.*(sum_g2ipi01-g25.*p5_01buf-I2)-lambda01.*g25;
            a1_01=beta1.*(P1H0.* P2H1).^2.*h1+h1.*beta01.*g21.^2;
            a2_01=beta2.*(P1H0.* P2H1).^2.*h2+h2.*beta01.*g22.^2;
            a3_01=beta3.*(P1H0.* P2H1).^2.*h3+h3.*beta01.*g23.^2;
            a4_01=beta4.*(P1H0.* P2H1).^2.*h4+h4.*beta01.*g24.^2;
            a5_01=beta5.*(P1H0.* P2H1).^2.*h5+h5.*beta01.*g25.^2;
            b1_01=h1.*(phi1_01+varphi1_01);
            b2_01=h2.*(phi2_01+varphi2_01);
            b3_01=h3.*(phi3_01+varphi3_01);
            b4_01=h4.*(phi4_01+varphi4_01);
            b5_01=h1.*(phi5_01+varphi5_01);
            c1_01=sigma1.*(phi1_01+varphi1_01)-h1.*P1H0.*P2H1;
            c2_01=sigma2.*(phi2_01+varphi2_01)-h2.*P1H0.*P2H1;
            c3_01=sigma3.*(phi3_01+varphi3_01)-h3.*P1H0.*P2H1;
            c4_01=sigma4.*(phi4_01+varphi4_01)-h4.*P1H0.*P2H1;
            c5_01=sigma5.*(phi5_01+varphi5_01)-h5.*P1H0.*P2H1;
            p1_01tilde=(-b1_01+sqrt(b1_01.^2-4.*a1_01.*c1_01)) ./(2.*a1_01);
            p2_01tilde=(-b2_01+sqrt(b2_01.^2-4.*a2_01.*c2_01)) ./(2.*a2_01);
            p3_01tilde=(-b3_01+sqrt(b3_01.^2-4.*a3_01.*c3_01)) ./(2.*a3_01);
            p4_01tilde=(-b4_01+sqrt(b4_01.^2-4.*a4_01.*c4_01)) ./(2.*a4_01);
            p5_01tilde=(-b5_01+sqrt(b5_01.^2-4.*a5_01.*c5_01)) ./(2.*a5_01);
            
            p1_01bar=(-b1_01+sqrt(b1_01.^2-4.*a1_01.*c1_01)) ./(2.*a1_01);
            p2_01bar=(-b2_01+sqrt(b2_01.^2-4.*a2_01.*c2_01)) ./(2.*a2_01);
            p3_01bar=(-b3_01+sqrt(b3_01.^2-4.*a3_01.*c3_01)) ./(2.*a3_01);
            p4_01bar=(-b4_01+sqrt(b4_01.^2-4.*a4_01.*c4_01)) ./(2.*a4_01);
            p5_01bar=(-b5_01+sqrt(b5_01.^2-4.*a5_01.*c5_01)) ./(2.*a5_01);
            
            p_01tildeVec=[p1_01tilde;p2_01tilde;p3_01tilde;p4_01tilde;p5_01tilde];
            p_01barVec=[p1_01bar;p2_01bar;p3_01bar;p4_01bar;p5_01bar];
            
            
            %5 个 pi01依次迭代
            for channelNum = 1:M
                
                Lpi0= lagrangePi01(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,1); %1 :ST
                Lpi= lagrangePi01(p_01tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,1); %1 :ST
                if b1_01(channelNum)^2-4*a1_01(channelNum)*c1_01(channelNum) < 0
                    p1_01(channelNum) = 0;
                elseif p_01tildeVec(1,channelNum) >= 0&& Lpi<=Lpi0
                    p1_01(channelNum) =  min(p_01tildeVec(1,channelNum),P1out(channelNum));
                elseif p_01barVec(1,channelNum) >= 0 && Lpi<=Lpi0
                    p1_01(channelNum) =  min(p_01tildeVec(1,channelNum),P1out(channelNum));
                else
                    p1_01(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                Lpi0= lagrangePi01(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,2); %1 :ST
                Lpi= lagrangePi01(p_01tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,2); %1 :ST
                if b2_01(channelNum)^2-4*a2_01(channelNum)*c2_01(channelNum) < 0
                    p2_01(channelNum) = 0;
                elseif p_01tildeVec(2,channelNum) >= 0&& Lpi<=Lpi0
                    p2_01(channelNum) =  min(p_01tildeVec(2,channelNum),P2out(channelNum));
                elseif p_01barVec(2,channelNum) >= 0 && Lpi<=Lpi0
                    p2_01(channelNum) =  min(p_01tildeVec(2,channelNum),P2out(channelNum));
                else
                    p2_01(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi01(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,3); %1 :ST
                Lpi= lagrangePi01(p_01tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,3); %1 :ST
                if b3_01(channelNum)^2-4*a3_01(channelNum)*c3_01(channelNum) < 0
                    p3_01(channelNum) = 0;
                elseif p_01tildeVec(3,channelNum) >= 0&& Lpi<=Lpi0
                    p3_01(channelNum) =  min(p_01tildeVec(3,channelNum),P3out(channelNum));
                elseif p_01barVec(3,channelNum) >= 0 && Lpi<=Lpi0
                    p3_01(channelNum) =  min(p_01tildeVec(3,channelNum),P3out(channelNum));
                else
                    p3_01(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi01(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,4); %1 :ST
                Lpi= lagrangePi01(p_01tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,4); %1 :ST
                if b4_01(channelNum)^2-4*a4_01(channelNum)*c4_01(channelNum) < 0
                    p4_01(channelNum) = 0;
                elseif p_01tildeVec(4,channelNum) >= 0&& Lpi<=Lpi0
                    p4_01(channelNum) =  min(p_01tildeVec(4,channelNum),P4out(channelNum));
                elseif p_01barVec(4,channelNum) >= 0 && Lpi<=Lpi0
                    p4_01(channelNum) =  min(p_01tildeVec(4,channelNum),P4out(channelNum));
                else
                    p4_01(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi01(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,5); %1 :ST
                Lpi= lagrangePi01(p_01tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda01,beta01,PavVec,channelNum,5); %1 :ST
                if b5_01(channelNum)^2-4*a5_01(channelNum)*c5_01(channelNum) < 0
                    p5_01(channelNum) = 0;
                elseif p_01tildeVec(5,channelNum) >= 0&& Lpi<=Lpi0
                    p5_01(channelNum) =  min(p_01tildeVec(5,channelNum),P5out(channelNum));
                elseif p_01barVec(5,channelNum) >= 0 && Lpi<=Lpi0
                    p5_01(channelNum) =  min(p_01tildeVec(5,channelNum),P5out(channelNum));
                else
                    p5_01(channelNum) = 0;
                end
            end
            %% pj10 子问题
            phi1_10=beta1.*P1H1.*P2H0.*(P1H0*P2H0.* p1_00buf + P1H0*P2H1.*p1_01buf + P1H1*P2H1.*p1_11buf-P1av)-lambda1.* P1H1.*P2H0;
            phi2_10=beta2.*P1H1.*P2H0.*(P1H0*P2H0.* p2_00buf + P1H0*P2H1.*p2_01buf + P1H1*P2H1.*p2_11buf-P2av)-lambda2.* P1H1.*P2H0;
            phi3_10=beta3.*P1H1.*P2H0.*(P1H0*P2H0.* p3_00buf + P1H0*P2H1.*p3_01buf + P1H1*P2H1.*p3_11buf-P3av)-lambda3.* P1H1.*P2H0;
            phi4_10=beta4.*P1H1.*P2H0.*(P1H0*P2H0.* p4_00buf + P1H0*P2H1.*p4_01buf + P1H1*P2H1.*p4_11buf-P4av)-lambda4.* P1H1.*P2H0;
            phi5_10=beta5.*P1H1.*P2H0.*(P1H0*P2H0.* p5_00buf + P1H0*P2H1.*p5_01buf + P1H1*P2H1.*p5_11buf-P5av)-lambda5.* P1H1.*P2H0;
            sum_g1ipi10=g11.*p1_10buf+g12.*p2_10buf+g13.*p3_10buf+g14.*p4_10buf+g15.*p5_10buf;
            varphi1_10=beta10.*g11.*(sum_g1ipi10-g11.*p1_10buf-I1)-lambda10.*g11;
            varphi2_10=beta10.*g12.*(sum_g1ipi10-g12.*p2_10buf-I1)-lambda10.*g12;
            varphi3_10=beta10.*g13.*(sum_g1ipi10-g13.*p3_10buf-I1)-lambda10.*g13;
            varphi4_10=beta10.*g14.*(sum_g1ipi10-g14.*p4_10buf-I1)-lambda10.*g14;
            varphi5_10=beta10.*g15.*(sum_g1ipi10-g15.*p5_10buf-I1)-lambda10.*g15;
            a1_10=beta1.*(P1H1.* P2H0).^2.*h1+h1.*beta10.*g11.^2;
            a2_10=beta2.*(P1H1.* P2H0).^2.*h2+h2.*beta10.*g12.^2;
            a3_10=beta3.*(P1H1.* P2H0).^2.*h3+h3.*beta10.*g13.^2;
            a4_10=beta4.*(P1H1.* P2H0).^2.*h4+h4.*beta10.*g14.^2;
            a5_10=beta5.*(P1H1.* P2H0).^2.*h5+h5.*beta10.*g15.^2;
            b1_10=h1.*(phi1_10+varphi1_10);
            b2_10=h2.*(phi2_10+varphi2_10);
            b3_10=h3.*(phi3_10+varphi3_10);
            b4_10=h4.*(phi4_10+varphi4_10);
            b5_10=h1.*(phi5_10+varphi5_10);
            c1_10=sigma1.*(phi1_10+varphi1_10)-h1.*P1H1.*P2H0;
            c2_10=sigma2.*(phi2_10+varphi2_10)-h2.*P1H1.*P2H0;
            c3_10=sigma3.*(phi3_10+varphi3_10)-h3.*P1H1.*P2H0;
            c4_10=sigma4.*(phi4_10+varphi4_10)-h4.*P1H1.*P2H0;
            c5_10=sigma5.*(phi5_10+varphi5_10)-h5.*P1H1.*P2H0;
            p1_10tilde=(-b1_10+sqrt(b1_10.^2-4.*a1_10.*c1_10)) ./(2.*a1_10);
            p2_10tilde=(-b2_10+sqrt(b2_10.^2-4.*a2_10.*c2_10)) ./(2.*a2_10);
            p3_10tilde=(-b3_10+sqrt(b3_10.^2-4.*a3_10.*c3_10)) ./(2.*a3_10);
            p4_10tilde=(-b4_10+sqrt(b4_10.^2-4.*a4_10.*c4_10)) ./(2.*a4_10);
            p5_10tilde=(-b5_10+sqrt(b5_10.^2-4.*a5_10.*c5_10)) ./(2.*a5_10);
            
            p1_10bar=(-b1_10+sqrt(b1_10.^2-4.*a1_10.*c1_10)) ./(2.*a1_10);
            p2_10bar=(-b2_10+sqrt(b2_10.^2-4.*a2_10.*c2_10)) ./(2.*a2_10);
            p3_10bar=(-b3_10+sqrt(b3_10.^2-4.*a3_10.*c3_10)) ./(2.*a3_10);
            p4_10bar=(-b4_10+sqrt(b4_10.^2-4.*a4_10.*c4_10)) ./(2.*a4_10);
            p5_10bar=(-b5_10+sqrt(b5_10.^2-4.*a5_10.*c5_10)) ./(2.*a5_10);
            
            p_10tildeVec=[p1_10tilde;p2_10tilde;p3_10tilde;p4_10tilde;p5_10tilde];
            p_10barVec=[p1_10bar;p2_10bar;p3_10bar;p4_10bar;p5_10bar];
            
            
            %5 个 pi10依次迭代
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi10(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,1); %1 :ST
                Lpi= lagrangePi10(p_10tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,1); %1 :ST
                if b1_10(channelNum)^2-4*a1_10(channelNum)*c1_10(channelNum) < 0
                    p1_10(channelNum) = 0;
                elseif p_10tildeVec(1,channelNum) >= 0&& Lpi<=Lpi0
                    p1_10(channelNum) =  min(p_10tildeVec(1,channelNum),P1out(channelNum));
                elseif p_10barVec(1,channelNum) >= 0 && Lpi<=Lpi0
                    p1_10(channelNum) =  min(p_10tildeVec(1,channelNum),P1out(channelNum));
                else
                    p1_10(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi10(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,2); %1 :ST
                Lpi= lagrangePi10(p_10tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,2); %1 :ST
                if b2_10(channelNum)^2-4*a2_10(channelNum)*c2_10(channelNum) < 0
                    p2_10(channelNum) = 0;
                elseif p_10tildeVec(2,channelNum) >= 0&& Lpi<=Lpi0
                    p2_10(channelNum) =  min(p_10tildeVec(2,channelNum),P2out(channelNum));
                elseif p_10barVec(2,channelNum) >= 0 && Lpi<=Lpi0
                    p2_10(channelNum) =  min(p_10tildeVec(2,channelNum),P2out(channelNum));
                else
                    p2_10(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi10(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,3); %1 :ST
                Lpi= lagrangePi10(p_10tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,3); %1 :ST
                if b3_10(channelNum)^2-4*a3_10(channelNum)*c3_10(channelNum) < 0
                    p3_10(channelNum) = 0;
                elseif p_10tildeVec(3,channelNum) >= 0&& Lpi<=Lpi0
                    p3_10(channelNum) =  min(p_10tildeVec(3,channelNum),P3out(channelNum));
                elseif p_10barVec(3,channelNum) >= 0 && Lpi<=Lpi0
                    p3_10(channelNum) =  min(p_10tildeVec(3,channelNum),P3out(channelNum));
                else
                    p3_10(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi10(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,4); %1 :ST
                Lpi= lagrangePi10(p_10tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,4); %1 :ST
                if b4_10(channelNum)^2-4*a4_10(channelNum)*c4_10(channelNum) < 0
                    p4_10(channelNum) = 0;
                elseif p_10tildeVec(4,channelNum) >= 0&& Lpi<=Lpi0
                    p4_10(channelNum) =  min(p_10tildeVec(4,channelNum),P4out(channelNum));
                elseif p_10barVec(4,channelNum) >= 0 && Lpi<=Lpi0
                    p4_10(channelNum) =  min(p_10tildeVec(4,channelNum),P4out(channelNum));
                else
                    p4_10(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi10(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,5); %5 :ST
                Lpi= lagrangePi10(p_10tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda10,beta10,PavVec,channelNum,5); %5 :ST
                if b5_10(channelNum)^2-4*a5_10(channelNum)*c5_10(channelNum) < 0
                    p5_10(channelNum) = 0;
                elseif p_10tildeVec(5,channelNum) >= 0&& Lpi<=Lpi0
                    p5_10(channelNum) =  min(p_10tildeVec(5,channelNum),P5out(channelNum));
                elseif p_10barVec(5,channelNum) >= 0 && Lpi<=Lpi0
                    p5_10(channelNum) =  min(p_10tildeVec(5,channelNum),P5out(channelNum));
                else
                    p5_10(channelNum) = 0;
                end
            end
            
            %% pj11 子问题
            phi1_11=beta1.*P1H1.*P2H1.*(P1H0*P2H0.* p1_00buf + P1H0*P2H1.*p1_01buf + P1H1*P2H0.*p1_10buf-P1av)-lambda1.* P1H1.*P2H1;
            phi2_11=beta2.*P1H1.*P2H1.*(P1H0*P2H0.* p2_00buf + P1H0*P2H1.*p2_01buf + P1H1*P2H0.*p2_10buf-P2av)-lambda2.* P1H1.*P2H1;
            phi3_11=beta3.*P1H1.*P2H1.*(P1H0*P2H0.* p3_00buf + P1H0*P2H1.*p3_01buf + P1H1*P2H0.*p3_10buf-P3av)-lambda3.* P1H1.*P2H1;
            phi4_11=beta4.*P1H1.*P2H1.*(P1H0*P2H0.* p4_00buf + P1H0*P2H1.*p4_01buf + P1H1*P2H0.*p4_10buf-P4av)-lambda4.* P1H1.*P2H1;
            phi5_11=beta5.*P1H1.*P2H1.*(P1H0*P2H0.* p5_00buf + P1H0*P2H1.*p5_01buf + P1H1*P2H0.*p5_10buf-P5av)-lambda5.* P1H1.*P2H1;
            sum_g1ipi11=g11.*p1_11buf+g12.*p2_11buf+g13.*p3_11buf+g14.*p4_11buf+g15.*p5_11buf;
            sum_g2ipi11=g21.*p1_11buf+g22.*p2_11buf+g23.*p3_11buf+g24.*p4_11buf+g25.*p5_11buf;
            varphi1_11=beta111.*g11.*(sum_g1ipi11-g11.*p1_11buf-I1)-lambda111.*g11 ...
                +beta112.*g21.*(sum_g2ipi11-g21.*p1_11buf-I2)-lambda112.*g21;
            varphi2_11=beta111.*g12.*(sum_g1ipi11-g12.*p2_11buf-I1)-lambda111.*g12 ...
                +beta112.*g22.*(sum_g2ipi11-g22.*p2_11buf-I2)-lambda112.*g22;
            varphi3_11=beta111.*g13.*(sum_g1ipi11-g13.*p3_11buf-I1)-lambda111.*g13 ...
                +beta112.*g23.*(sum_g2ipi11-g23.*p3_11buf-I2)-lambda112.*g23;
            varphi4_11=beta111.*g14.*(sum_g1ipi11-g14.*p4_11buf-I1)-lambda111.*g14 ...
                +beta112.*g24.*(sum_g2ipi11-g24.*p4_11buf-I2)-lambda112.*g24;
            varphi5_11=beta111.*g15.*(sum_g1ipi11-g15.*p5_11buf-I1)-lambda111.*g15 ...
                +beta112.*g25.*(sum_g2ipi11-g25.*p5_11buf-I2)-lambda112.*g25;
            a1_11=beta1.*(P1H1.* P2H1).^2.*h1+h1.*beta111.*g11.^2+h1.*beta112.*g21.^2;
            a2_11=beta2.*(P1H1.* P2H1).^2.*h2+h2.*beta111.*g12.^2+h2.*beta112.*g22.^2;
            a3_11=beta3.*(P1H1.* P2H1).^2.*h3+h3.*beta111.*g13.^2+h3.*beta112.*g23.^2;
            a4_11=beta4.*(P1H1.* P2H1).^2.*h4+h4.*beta111.*g14.^2+h4.*beta112.*g24.^2;
            a5_11=beta5.*(P1H1.* P2H1).^2.*h5+h5.*beta111.*g15.^2+h5.*beta112.*g25.^2;
            b1_11=h1.*(phi1_11+varphi1_11);
            b2_11=h2.*(phi2_11+varphi2_11);
            b3_11=h3.*(phi3_11+varphi3_11);
            b4_11=h4.*(phi4_11+varphi4_11);
            b5_11=h1.*(phi5_11+varphi5_11);
            c1_11=sigma1.*(phi1_11+varphi1_11)-h1.*P1H1.* P2H1;
            c2_11=sigma2.*(phi2_11+varphi2_11)-h2.*P1H1.* P2H1;
            c3_11=sigma3.*(phi3_11+varphi3_11)-h3.*P1H1.* P2H1;
            c4_11=sigma4.*(phi4_11+varphi4_11)-h4.*P1H1.* P2H1;
            c5_11=sigma5.*(phi5_11+varphi5_11)-h5.*P1H1.* P2H1;
            p1_11tilde=(-b1_11+sqrt(b1_11.^2-4.*a1_11.*c1_11)) ./(2.*a1_11);
            p2_11tilde=(-b2_11+sqrt(b2_11.^2-4.*a2_11.*c2_11)) ./(2.*a2_11);
            p3_11tilde=(-b3_11+sqrt(b3_11.^2-4.*a3_11.*c3_11)) ./(2.*a3_11);
            p4_11tilde=(-b4_11+sqrt(b4_11.^2-4.*a4_11.*c4_11)) ./(2.*a4_11);
            p5_11tilde=(-b5_11+sqrt(b5_11.^2-4.*a5_11.*c5_11)) ./(2.*a5_11);
            
            p1_11bar=(-b1_11+sqrt(b1_11.^2-4.*a1_11.*c1_11)) ./(2.*a1_11);
            p2_11bar=(-b2_11+sqrt(b2_11.^2-4.*a2_11.*c2_11)) ./(2.*a2_11);
            p3_11bar=(-b3_11+sqrt(b3_11.^2-4.*a3_11.*c3_11)) ./(2.*a3_11);
            p4_11bar=(-b4_11+sqrt(b4_11.^2-4.*a4_11.*c4_11)) ./(2.*a4_11);
            p5_11bar=(-b5_11+sqrt(b5_11.^2-4.*a5_11.*c5_11)) ./(2.*a5_11);
            
            p_11tildeVec=[p1_11tilde;p2_11tilde;p3_11tilde;p4_11tilde;p5_11tilde];
            p_11barVec=[p1_11bar;p2_11bar;p3_11bar;p4_11bar;p5_11bar];
            
            
            %5 个 pi11依次迭代
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi11(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,1); %1 :ST
                Lpi= lagrangePi11(p_11tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,1); %1 :ST
                if b1_11(channelNum)^2-4*a1_11(channelNum)*c1_11(channelNum) < 0
                    p1_11(channelNum) = 0;
                elseif p_11tildeVec(1,channelNum) >= 0&& Lpi<=Lpi0
                    p1_11(channelNum) = min(p_11tildeVec(1,channelNum),P1out(channelNum));
                elseif p_11barVec(1,channelNum) >= 0 && Lpi<=Lpi0
                    p1_11(channelNum) =  min(p_11tildeVec(1,channelNum),P1out(channelNum));
                else
                    p1_11(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi11(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,2); %1 :ST
                Lpi= lagrangePi11(p_11tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,2); %1 :ST
                if b2_11(channelNum)^2-4*a2_11(channelNum)*c2_11(channelNum) < 0
                    p2_11(channelNum) = 0;
                elseif p_11tildeVec(2,channelNum) >= 0&& Lpi<=Lpi0
                    p2_11(channelNum) =  min(p_11tildeVec(2,channelNum),P2out(channelNum));
                elseif p_11barVec(2,channelNum) >= 0 && Lpi<=Lpi0
                    p2_11(channelNum) = min(p_11tildeVec(2,channelNum),P2out(channelNum));
                else
                    p2_11(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi11(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,3); %1 :ST
                Lpi= lagrangePi11(p_11tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,3); %1 :ST
                if b3_11(channelNum)^2-4*a3_11(channelNum)*c3_11(channelNum) < 0
                    p3_11(channelNum) = 0;
                elseif p_11tildeVec(3,channelNum) >= 0&& Lpi<=Lpi0
                    p3_11(channelNum) = min(p_11tildeVec(3,channelNum),P3out(channelNum));
                elseif p_11barVec(3,channelNum) >= 0 && Lpi<=Lpi0
                    p3_11(channelNum) = min(p_11tildeVec(3,channelNum),P3out(channelNum));
                else
                    p3_11(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi11(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,4); %1 :ST
                Lpi= lagrangePi11(p_11tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,4); %1 :ST
                if b4_11(channelNum)^2-4*a4_11(channelNum)*c4_11(channelNum) < 0
                    p4_11(channelNum) = 0;
                elseif p_11tildeVec(4,channelNum) >= 0&& Lpi<=Lpi0
                    p4_11(channelNum) = min(p_11tildeVec(4,channelNum),P4out(channelNum));
                elseif p_11barVec(4,channelNum) >= 0 && Lpi<=Lpi0
                    p4_11(channelNum) = min(p_11tildeVec(4,channelNum),P4out(channelNum));
                else
                    p4_11(channelNum) = 0;
                end
            end
            for channelNum = 1:M
                
                
                Lpi0= lagrangePi11(zeros(5,M),p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,5); %1 :ST
                Lpi= lagrangePi11(p_11tildeVec,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                    ,PHvec,gVec1,gVec2,hVec,I1,I2,lambdaVec,betaVec,lambda111,beta111,lambda112,beta112,PavVec,channelNum,5); %1 :ST
                if b5_11(channelNum)^2-4*a5_11(channelNum)*c5_11(channelNum) < 0
                    p5_11(channelNum) = 0;
                elseif p_11tildeVec(5,channelNum) >= 0&& Lpi<=Lpi0
                    p5_11(channelNum) = min(p_11tildeVec(5,channelNum),P5out(channelNum));
                elseif p_11barVec(5,channelNum) >= 0 && Lpi<=Lpi0
                    p5_11(channelNum) = min(p_11tildeVec(5,channelNum),P5out(channelNum));
                else
                    p5_11(channelNum) = 0;
                end
            end
            
            
            
            lambda1 = lambda1-beta1.*(P1H0*P2H0.* p1_00 + P1H0*P2H1.*p1_01 + P1H1*P2H0.*p1_10 + P1H1*P2H1.*p1_11-P1av);
            lambda2 = lambda2-beta2.*(P1H0*P2H0.* p2_00 + P1H0*P2H1.*p2_01 + P1H1*P2H0.*p2_10 + P1H1*P2H1.*p2_11-P2av);
            lambda3 = lambda3-beta3.*(P1H0*P2H0.* p3_00 + P1H0*P2H1.*p3_01 + P1H1*P2H0.*p3_10 + P1H1*P2H1.*p3_11-P3av);
            lambda4 = lambda4-beta4.*(P1H0*P2H0.* p4_00 + P1H0*P2H1.*p4_01 + P1H1*P2H0.*p4_10 + P1H1*P2H1.*p4_11-P4av);
            lambda5 = lambda5-beta5.*(P1H0*P2H0.* p5_00 + P1H0*P2H1.*p5_01 + P1H1*P2H0.*p5_10 + P1H1*P2H1.*p5_11-P5av);
            
            
            lambda01= lambda01 - beta01.*(g21.*p1_01+g22.*p2_01+g23.*p3_01+g24.*p4_01+g25.*p5_01-I2);
            
            lambda10= lambda10 - beta10.*(g11.*p1_10+g12.*p2_10+g13.*p3_10+g14.*p4_10+g15.*p5_10-I1);
            
            lambda111= lambda111 - beta111.*(g11.*p1_11+g12.*p2_11+g13.*p3_11+g14.*p4_11+g15.*p5_11-I1);
            lambda112= lambda112 - beta112.*(g21.*p1_11+g22.*p2_11+g23.*p3_11+g24.*p4_11+g25.*p5_11-I2);
           
            EC1_00(nn) = mean(log(1+h1.*p1_00));
        EC2_00(nn) = mean(log(1+h2.*p2_00));
        EC3_00(nn) = mean(log(1+h3.*p3_00));
        EC4_00(nn) = mean(log(1+h4.*p4_00));
        EC5_00(nn) = mean(log(1+h5.*p5_00));
        ECsum_00(nn) = mean(log(1+h1.*p1_00)+log(1+h2.*p2_00)+log(1+h3.*p3_00)+log(1+h4.*p4_00)+log(1+h5.*p5_00));
        EC1_01(nn) = mean(log(1+h1.*p1_01));
        EC2_01(nn) = mean(log(1+h2.*p2_01));
        EC3_01(nn) = mean(log(1+h3.*p3_01));
        EC4_01(nn) = mean(log(1+h4.*p4_01));
        EC5_01(nn) = mean(log(1+h5.*p5_01));
        ECsum_01(nn) = mean(log(1+h1.*p1_01)+log(1+h2.*p2_01)+log(1+h3.*p3_01)+log(1+h4.*p4_01)+log(1+h5.*p5_01));
        EC1_10(nn) = mean(log(1+h1.*p1_10));
        EC2_10(nn) = mean(log(1+h2.*p2_10));
        EC3_10(nn) = mean(log(1+h3.*p3_10));
        EC4_10(nn) = mean(log(1+h4.*p4_10));
        EC5_10(nn) = mean(log(1+h5.*p5_10));
        ECsum_10(nn) = mean(log(1+h1.*p1_10)+log(1+h2.*p2_10)+log(1+h3.*p3_10)+log(1+h4.*p4_10)+log(1+h5.*p5_10));
        EC1_11(nn) = mean(log(1+h1.*p1_11));
        EC2_11(nn) = mean(log(1+h2.*p2_11));
        EC3_11(nn) = mean(log(1+h3.*p3_11));
        EC4_11(nn) = mean(log(1+h4.*p4_11));
        EC5_11(nn) = mean(log(1+h5.*p5_11));
        ECsum_11(nn) = mean(log(1+h1.*p1_11)+log(1+h2.*p2_11)+log(1+h3.*p3_11)+log(1+h4.*p4_11)+log(1+h5.*p5_11));
        ECsum(nn)=P1H0*P2H0 * ECsum_00(nn) + P1H0*P2H1*ECsum_01(nn) + P1H1*P2H0 * ECsum_10(nn) + P1H1*P2H1 * ECsum_11(nn);
        end
%             Delta1_00 = norm(p1_00-p1_00buf)/norm(p1_00buf);
%             Delta1_01 = norm(p1_01-p1_01buf)/norm(p1_01buf);
%             Delta1_10 = norm(p1_10-p1_10buf)/norm(p1_10buf);
%             Delta1_11 = norm(p1_11-p1_11buf)/norm(p1_11buf);
%             
%             Delta2_00 = norm(p2_00-p2_00buf)/norm(p2_00buf);
%             Delta2_01 = norm(p2_01-p2_01buf)/norm(p2_01buf);
%             Delta2_10 = norm(p2_10-p2_10buf)/norm(p2_10buf);
%             Delta2_11 = norm(p2_11-p2_11buf)/norm(p2_11buf);
%             
%             Delta3_00 = norm(p3_00-p3_00buf)/norm(p3_00buf);
%             Delta3_01 = norm(p3_01-p3_01buf)/norm(p3_01buf);
%             Delta3_10 = norm(p3_10-p3_10buf)/norm(p3_10buf);
%             Delta3_11 = norm(p3_11-p3_11buf)/norm(p3_11buf);
%             
%             Delta4_00 = norm(p4_00-p4_00buf)/norm(p4_00buf);
%             Delta4_01 = norm(p4_01-p4_01buf)/norm(p4_01buf);
%             Delta4_10 = norm(p4_10-p4_10buf)/norm(p4_10buf);
%             Delta4_11 = norm(p4_11-p4_11buf)/norm(p4_11buf);
%             
%             Delta5_00 = norm(p5_00-p5_00buf)/norm(p5_00buf);
%             Delta5_11 = norm(p5_11-p5_11buf)/norm(p5_11buf);
%             Delta5_10 = norm(p5_10-p5_10buf)/norm(p5_10buf);
%             Delta5_01 = norm(p5_01-p5_01buf)/norm(p5_01buf);
%             
%             if      Delta1_00 < Epsilon1 && Delta1_01 < Epsilon1 &&Delta1_10 < Epsilon1 &&Delta1_11 < Epsilon1 &&...
%                     Delta2_00 < Epsilon2 && Delta2_01 < Epsilon2 &&Delta2_10 < Epsilon2 &&Delta2_11 < Epsilon2 &&...
%                     Delta3_00 < Epsilon3 && Delta3_01 < Epsilon3 &&Delta3_10 < Epsilon3 &&Delta3_11 < Epsilon3 &&...
%                     Delta4_00 < Epsilon4 && Delta4_01 < Epsilon4 &&Delta4_10 < Epsilon4 &&Delta4_11 < Epsilon4 &&...
%                     Delta5_00 < Epsilon5 && Delta5_01 < Epsilon5 &&Delta5_10 < Epsilon5 &&Delta5_11 < Epsilon5
%                 break;
%             end

%end                
    % EC1gao(ii,:)=EC1(:);
    % EC2gao(ii,:)=EC1(:);
    % EC3gao(ii,:)=EC1(:);
%     ECsumgao(jj,:)=ECsum(:);
%     jj=jj+1;
% end
ECsumMat(iii+1,:)=ECsum;
        end
figure
line = plot(1:NN,ECsumMat(1,:)*B,'b:',1:NN,ECsumMat(2,:)*B,...
    'g-.',1:NN,ECsumMat(3,:)*B,'r-');

set(line,'MarkerSize',8,'LineWidth',2);
grid;
legend('ADMM,\beta=0.01','ADMM,\beta=1','ADMM,\beta=100');
xlabel('Iter (k)');
ylabel('Ergodic Capacity (bits/s/Hz)');
ADMMbeta=ECsumMat.*B;
save  ADMMbeta
% EtaNOsensing=ECsumgao;
% save EtaNOsensing
% axis([-20 10 0 6 ]);



