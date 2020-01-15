n = [ 200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000];
%k=10
seq10 = [0.06 0.241 0.5389 0.957 1.5 2.15 2.921 3.82 4.826 5.97 7.21 8.57 10.067 11.67 13.5 15.6 17.33471 19.45 21.675 24];
v1_10 = [0.0367 0.0465 0.0563 0.0747 0.1369 0.162 0.151 0.1867 0.24 0.267 0.34 0.3667 0.45 0.51 0.59 0.6281 0.76 0.85 0.93 1.038];
v2_10 = [ 0.04021 0.0451 0.0558 0.074 0.102 0.1264 0.1521 0.1824 0.2234 0.2661 0.3167 0.3545 0.41 0.4702 0.5525 0.6140 0.6766 0.753 0.8427 0.937];
v3_10 = [ 0.0379 0.044 0.0457 0.0537 0.06 0.0755 0.0833 0.096 0.1092 0.1239 0.15 0.167 0.187 0.2057 0.2276 0.2665 0.2888 0.3112 0.344 0.371]; 
%k=50
seq50 = [ 0.2556 0.1016 2.281 4.058 6.3297 9.1225 12.41 16.25 20.5491 25.3 32.6219 36.43 42.7914 49.6166 57.6397 65.5608 74.2594 83.0628 92.55 102.3798];
v1_50 = [ 0.0495 0.078 0.128 0.212 0.3258 0.41 0.51 0.65 0.81 1.02 1.287 1.4385 1.6789 1.9 2.21 2.5322 2.83 3.14 3.46 3.8498];
v2_50 = [ 0.0496 0.072 0.117 0.19 0.2644 0.3962 0.464 0.62 0.8 1 1.211 1.3729 1.62 1.8844 2.1558 2.467 2.77 3.08 3.43 3.7938]; 
v3_50 = [ 0.0496 0.0605 0.083 0.1067 0.1367 0.2059 0.255 0.3111 0.38 0.445 0.5827 0.6702 0.75 0.8636 0.98 1.166 1.28 1.415 1.5542 1.66];
%k=100
seq100 = [ 0.3516 1.985 4.5 7.925 13.37 17.81 24.275 31.6482 40.05 49.4597 59.8284 71.2053 83.554 96.91 112.5057 128.5 144.5783 162.4 180.6623 200.4158];
v1_100 = [ 0.063 0.124 0.21 0.36 0.52 0.74 0.995 1.243 1.52 1.855 2.29 2.7067 3.011 3.55 4.148 4.76 5.28 5.956 6.6 7.256];
v2_100 = [ 0.067 0.11 0.191 0.308 0.483 0.6694 0.8708 1.1078 1.408 1.775 2.1426 2.5048 2.92 3.389 3.973 4.5312 5.075 5.659 6.354 7.088];
v3_100 = [ 0.06 0.086 0.1305 0.1845 0.2445 0.3866 0.4749 0.5849 0.716 0.85 1.074 1.237 1.435 1.607 1.8185 2.168 2.425 2.64 2.912 3.2];

%sequential 
figure
plot( n ,seq10, 'DisplayName' , 'k=10')
title('\fontsize{20} Sequential response')
ylabel('\fontsize{20} time(sec)')
xlabel('\fontsize{20} Parameter n')
set(gca,'FontSize',20)
hold on
plot(n, seq50 , 'DisplayName' , 'k=50' )
plot(n, seq100 , 'DisplayName' , 'k=100')
legend('\fontsize{32} k=10','\fontsize{32} k=50','\fontsize{32} k=100');

%V1 
figure
plot( n ,v1_10, 'DisplayName' , 'k=10')
title('\fontsize{20} V1 response')
ylabel('\fontsize{20} time(sec)')
xlabel('\fontsize{20} Parameter n')
set(gca,'FontSize',20)
hold on
plot(n, v1_50, 'DisplayName' , 'k=50' )
plot(n, v1_100 , 'DisplayName' , 'k=100')
legend('\fontsize{32} k=10','\fontsize{32} k=50','\fontsize{32} k=100');

%V2 
figure
plot( n ,v2_10, 'DisplayName' , 'k=10')
title('\fontsize{20} V2 response')
ylabel('\fontsize{20} time(sec)')
xlabel('\fontsize{20} Parameter n')
set(gca,'FontSize',20)
hold on
plot(n, v2_50, 'DisplayName' , 'k=50' )
plot(n, v2_100 , 'DisplayName' , 'k=100')
legend('\fontsize{32} k=10','\fontsize{32} k=50','\fontsize{32} k=100');

%V3 
figure
plot( n ,v3_10, 'DisplayName' , 'k=10')
title('\fontsize{20} V3 response')
ylabel('\fontsize{20} time(sec)')
xlabel('\fontsize{20} Parameter n')
set(gca,'FontSize',20)
hold on
plot(n, v3_50, 'DisplayName' , 'k=50' )
plot(n, v3_100 , 'DisplayName' , 'k=100')
legend('\fontsize{32} k=10','\fontsize{32} k=50','\fontsize{32} k=100');

%All together
figure
plot( n ,seq100, 'DisplayName' , 'seq')
title('\fontsize{20} Parallel Vs Sequential k=100')
ylabel('\fontsize{20} time(sec)')
xlabel('\fontsize{20} Parameter n')
set(gca,'FontSize',20)
hold on
plot(n, v1_100, 'DisplayName' , 'v1' )
plot(n, v2_100 , 'DisplayName' , 'v2')
plot(n, v3_100 , 'DisplayName' , 'v3')
legend('\fontsize{32} seq','\fontsize{32} v1','\fontsize{32} v2' ,'\fontsize{32} v3');

%Parallel comparison
figure
plot(n, v1_100, 'DisplayName' , 'v1' )
title('\fontsize{20} Response comparison between parallel implementations for k=100')
ylabel('\fontsize{20} time(sec)')
xlabel('\fontsize{20} Parameter n')
set(gca,'FontSize',20)
hold on
plot(n, v2_100 , 'DisplayName' , 'v2')
plot(n, v3_100 , 'DisplayName' , 'v3')
legend('\fontsize{32} v1','\fontsize{32} v2' ,'\fontsize{32} v3');

