    function  L= lagrangePi01(p_Variable,p_00Vec,p_01Vec,p_10Vec,p_11Vec...
                ,PHvec,g1Vec,g2Vec,hVec,I1,I2,lambdaVec,betaVec,lambda01Vec,beta01,PavVec,channelNum,ST)
            %I1 是主用户PT 1 的上限干扰功率
    % k 只是lambda的迭代K次的一个表示
    % N=5次级用户数目PT,2主用户PR，4是00 01 10 11 
    %channelNuM_i 通道实现 第i个 共有1e4个
    %ST_j 次级传输用户 j
    %p_00Vec=[p1_00;p2_00;p3_00;p4_00;p5_00]
    %p_01Vec=[p1_01;p2_01;p3_01;p4_01;p5_01]
    %p_10Vec=[p1_10;p2_10;p3_10;p4_10;p5_10]
    %p_11Vec=[p1_11;p2_11;p3_11;p4_11;p5_11]
    %PHvec=[P1H0,P1H1,P2H0,P2H1]
    %gVec1=[g11;g12;g13;g14;g15];针对PR 1 的 5 ST 干扰gain
    %gVec2=[g21;g22;g23;g24;g25];针对PR 2 的 5 ST 干扰gain
    %hVec=[h1;h2;h3;h4;h5]; 5 ST 的发射gain
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
    Pav=PavVec(ST);% 对次级用户ST，信道实现序列channelNum，其上限功率限制为Pav（ST，channelNum）
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