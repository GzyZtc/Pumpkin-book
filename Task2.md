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

<h3>解析解</h3>

基于均方误差最小化来进行模型求解的方法称作**最小二乘法(least square method)**,在线性回归中，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之和最小

求解$\omega,b$,使得

$$E_(\omega,b)=\underset{i=1}{\overset{m}{\sum}}(y_i-\omega x_i-b)^2$$

最小化的过程，成为线性回归的最小二乘**参数估计(pamameter estimation)**。对$E_{(\omega,b)}$分别对$\omega$和$b$求导，得到

$$\frac{\partial E_{(\omega,b)}}{\partial \omega}=2(\omega\underset{i=1}{\overset{m}{\sum}}x_i^2-\underset{i=1}{\overset{m}{\sum}}(y_i-b)x_i)$$

$$\frac{\partial E_{(\omega,b)}}{\partial \omega}=2(mb-\underset{i=1}{\overset{m}{\sum}}(y_i-\omega x_i))$$

让偏导数等于零求得极值点，可得到$\omega,b$最优解的闭式解

$$\omega = \frac{\underset{i=1}{\overset{m}{\sum}}y_i(x_i-\overset{-}{x})}{\underset{i=1}{\overset{m}{\sum}}x^2_i-\frac{1}{m}(\underset{i=1}{\overset{m}{\sum}}x_i)^2}$$

$$b=\frac{1}{m}\underset{i=1}{\overset{m}{\sum}}(y_i-\omega x_i)$$



对更一般的情形，我们试图学得

$$f(x_i)=\omega^{T}x_i+b,使得f(x_i)\simeq y_i$$

这成为**多元线性回归(multivariate linear regression)**

为便于讨论，我们把$\omega$和b写成向量形式

$$\hat{\omega}^{\ast}=(\omega ; b)$$

相应的，把数据集$D$表示为一个$m\times (d+1)$大小的矩阵$X$,其中每行对应一个示例，该行前d个元素对应于示例的d个属性值，最后一个元素恒置为1，即

$$X=\begin{pmatrix}
x_{11}&x_{12}&\cdots&x_{1d}&1\\
x_{21}&x_{22}&\cdots&x_{2d}&1\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
x_{m1}&x_{m2}&\cdots&x_{md}&1
\end{pmatrix}=\begin{pmatrix}
x_{1}^{T}&1\\
x_{2}^{T}&1\\
\vdots&\vdots\\ 
x_{m}^{T}&1
\end{pmatrix}$$

再把标记也写成向量形式 $y=(y_1;y_2;...;y_m)$,有

$$\hat{\omega}^{\ast}=\underset{\hat{\omega}}{argmin}(y-X\hat{\omega})^{T}(y-X\hat{\omega})$$

令

$$E_{\hat{\omega}}=(y-X\hat{\omega})^{T}(y-X\hat{\omega})$$

对$\hat{\omega}$求导得到

$$\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}}=2X^{T}(X\hat{\omega}-y)$$

令上式为0可得$\hat{\omega}$的闭式解,当$X^{T}X$满秩时,得

$$\hat{\omega}^{\ast}=(X^{T}X)^{-1}X^{T}y$$

令$\hat{x_{i}}=(x_{i},1)$,最终学得得多元线性模型为

$$f(\hat{x_{i}})=\hat{x_{i}}(X^{T}X)^{-1}X^{T}y$$

若$X^{T}X$不满秩,则对于矩阵方程可能有多个解$\hat{\omega}$,他们都能使得均方差无最小化,引入正则化项决定学习算法的归纳偏好

<h3>正则化</h3>

模型选择的典型方法是正则化(regularization)。正则化是结构风险最小化策略的实现，是在经验风险上加一个正则化项(regularizer)或罚项(penalty term)。正则化一般是模型复杂度的单调递增，模型越复杂，正则化就越大。

正则化一般具有以下形式

$$\underset{f\in \mathcal{F}}{min}\overset{N}{\underset{i=1}{\sum}}L(y_i,f(x_i))+\lambda J(f)$$

正则化项可以取不同的形式，在回归问题中，损失函数是平方损失，正则化项可以是参数向量$L_2$的范数：

$$L(\omega)=\frac{1}{N}\overset{N}{\underset{i=1}{\sum}}(f(x_{i};w)-y_i)^2+\frac{\lambda}{2}\|\omega \|_1$$

这里$\|\omega \|_{1}$ 表示参数向量 $\omega$的$L_{1}$范数

<br>
像线性回归这样的简单问题存在解析解，但不是所有的问题都存在解析解，解析解可以很好的进行数学分析，但解析解对问题的限制很严格。

<h3>随机梯度下降</h3>
即使在我们⽆法得到解析解的情况下，我们仍然可以有效地训练模型。在许多任务上，那些难以优化的模型
效果要更好。因此，弄清楚如何训练这些难以优化的模型是⾮常重要的。<br>



<h2>对数几率回归
<h2>LDA

