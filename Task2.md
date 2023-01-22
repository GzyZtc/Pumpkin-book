<h1>第三章 线性模型</h1>
<h2>线性回归</h2>

给定数据集

$$D = \{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$$

其中

$$x_i=(x_{i1};_{i2};...;x_{id}),y_i \in \mathbb{R}$$

线性回归试图学得

$$f(x_i)=\omega x_i + b,使得f(x_i)\simeq y_i$$

其中$\omega,b$使得均方误差最小化

$$(\omega ^{\ast},b^{\ast})=\underset{(\omega ,b)}{argmin}\overset{m}{\underset{i=1}{\sum}}(f(x_i)-y_i)^2=\underset{(\omega ,b)}{argmin }\overset{m}{\underset{i=1}{\sum}}(\omega x_i+b_i -y_i)^2
$$

基于均方误差最小化来进行模型求解的方法称作**最小二乘法(least square method)**,在线性回归中，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之和最小

求解$\omega,b$,使得

$$E_(\omega,b)=\underset{i=1}{\overset{m}{\sum}}(y_i-\omega x_i-b)^2$$

最小化的过程，成为线性回归的最小二乘**参数估计(pamameter estimation)**。对$E_{(\omega,b)}$分别对$\omega$和$b$求导，得到

$$\frac{\partial E_{(\omega,b)}}{\partial \omega}=2(\omega\underset{i=1}{\overset{m}{\sum}}x_i^2-\underset{i=1}{\overset{m}{\sum}}(y_i-b)x_i)$$

$$\frac{\partial E_{(\omega,b)}}{\partial \omega}=2(mb-\underset{i=1}{\overset{m}{\sum}}(y_i-\omega x_i))$$

让偏导数等于零求得极值点，可得到$\omega,b$最优解的闭式解

$$\omega = \frac{\underset{i=1}{\overset{m}{\sum}}y_i(x_i-\overset{-}{x})}{\underset{i=1}{\overset{m}{\sum}}x^2_i-\frac{1}{m}(\underset{i=1}{\overset{m}{\sum}}x_i)^2}$$

$$b=\frac{1}{m}\underset{i=1}{\overset{m}{\sum}}(y_i-\omega x_i)$$



<h2>对数几率回归
<h2>LDA
