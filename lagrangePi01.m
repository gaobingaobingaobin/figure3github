    function  L= lagrangePi01(p_Variable,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                ,PHvec,g1Vec,g2Vec,hVec,I1,I2,lambdaVec,betaVec,lambda01Vec,beta01,PavVec,channelNum,ST)
            %I1 �����û�PT 1 �����޸��Ź���
    % k ֻ��lambda�ĵ���K�ε�һ����ʾ
    % N=5�μ��û���ĿPT,2���û�PR��4��00 01 10 11 
    %channelNuM_i ͨ��ʵ�� ��i�� ����1e4��
    %ST_j �μ������û� j
    %p_00Vec=[p1_00;p2_00;p3_00;p4_00;p5_00]
    %p_01Vec=[p1_01;p2_01;p3_01;p4_01;p5_01]
    %p_10Vec=[p1_10;p2_10;p3_10;p4_10;p5_10]
    %p_11Vec=[p1_11;p2_11;p3_11;p4_11;p5_11]
    %PHvec=[P1H0,P1H1,P2H0,P2H1]
    %gVec1=[g11;g12;g13;g14;g15];���PR 1 �� 5 ST ����gain
    %gVec2=[g21;g22;g23;g24;g25];���PR 2 �� 5 ST ����gain
    %hVec=[h1;h2;h3;h4;h5]; 5 ST �ķ���gain
lambda=lambdaVec(ST,channelNum);
    beta=betaVec(ST);
    lambda01=lambda01Vec(channelNum);
%     beta01=beta01Vec(ST);
    p_00=p_00Vec(ST,channelNum);
    p_01=p_01Vec(ST,channelNum);
    p_10=p_10Vec(ST,channelNum);
    p_11=p_11Vec(ST,channelNum);
    h=hVec(ST,channelNum);
%     g1=g1Vec(ST,channelNum);
    g2=g2Vec(ST,channelNum);
    p_variable=p_Variable(ST,channelNum);
    Pav=PavVec(ST);% �Դμ��û�ST���ŵ�ʵ������channelNum�������޹�������ΪPav��ST��channelNum��
    P1H0=PHvec(1);
    P1H1=PHvec(2);
    P2H0=PHvec(3);
    P2H1=PHvec(4);
    tempSUM=P1H0.*P2H0.*p_00+P1H0.*P2H1.*p_01+P1H1.*P2H0.*p_10+...
           P1H1.*P2H1.*p_11;
       tempSUM2=sum(g2Vec.*p_01Vec);
    L = -P1H0.*P2H1.*log(1+h.*p_variable)-lambda.*P1H0.*P2H1.*p_variable...
        +beta./2 .* (tempSUM-P1H0.*P2H1.*p_01 +P1H0.*P2H1.*p_variable-Pav ).^2 ...
        -lambda01.*g2.*p_variable+beta01./2.* (tempSUM2(channelNum)-g2.*p_01+g2.*p_variable -I2).^2;

    end