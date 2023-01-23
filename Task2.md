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
即使在我们无法得到解析解的情况下，我们仍然可以有效地训练模型。在许多任务上，那些难以优化的模型效果要更好。因此，弄清楚如何训练这些难以优化的模型是非常重要的。<br>

梯度下降最简单的⽤法是计算损失函数（数据集中所有样本的损失均值）关于模型参数的导数（在这⾥也可以称为梯度）。在每次迭代中，我们首先随机抽样⼀个小批量$\beta$，它是由固定数量的训练样本组成的。然后，我们计算小批量的平均损失关于模型参数的导数（也可以称为梯度）。最后，我们将梯度乘以⼀个预先确定的正数η，并从
当前参数的值中减掉。

我们用下面的数学公式来表示这一更新过程

$$(\omega,b)←(\omega,b)-\frac{\eta}{|\mathcal{B}|}\underset{i\in \mathcal{B}}{\sum}\partial_{\omega,b}l^{(i)}(\omega,b)$$

其算法的步骤如下:
<ol>
<li>初始化模型参数的值,如随机初始化</li>
<li>从数据集中随机抽取小批样且在负梯度的方向上更新参数,且不断迭代这一步骤</li>
</ol>

对于平方损失和仿射变换,我们可以写成如下形式:

$$\omega \leftarrow\omega-\frac{\eta}{|\mathcal{B}|}\underset{i\in\mathcal{B}}{\sum}\partial_{\omega}l^{(i)}(\omega,b)=\omega-\frac{\eta}{|\mathcal{B}|}\underset{i\in\mathcal{B}}{\sum}x^{(i)}(\omega^{T}x^{(i)}+b-y^{(i)})$$

$$b \leftarrow b-\frac{\eta}{|\mathcal{B}|}\underset{i\in\mathcal{B}}{\sum}\partial_{b}l^{(i)}(\omega,b)=b-\frac{\eta}{|\mathcal{B}|}\underset{i\in\mathcal{B}}{\sum}x^{(i)}(\omega^{T}x^{(i)}+b-y^{(i)})$$


上述公式中,$\omega$和$x$都是向量,$|\mathcal{B}|$表示每个小批量中的样本数,称为批量大小(bath size),$\eta$表示学习率(learning rate).

在训练了若干迭代次数后,我们记录模型参数的估计值,表示为$\hat{\omega},\hat{b}$,但是即使我们的函数是线性的且无噪声的,这些估计值也不会使得损失函数真正地达到最小值.因为算法会使得损失向最小值缓慢收敛,但却不能在有限的步数内非常精确地达到最小值.

<h3>正态分布与平方损失</h3>

正态分布和线性回归之间的关系很密切。正态分布（normal distribution），也称为高斯分布（Gaussian distribution），最早由德国数学家高斯（Gauss）应⽤于天文学研究。简单的说，若随机变量x具有均值µ和方差σ2（标准差σ），其正态分布概率密度函数如下：

$$p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{1}{2\sigma^2}(x-\mu)^2)$$

均方误差损失函数（简称均方损失）可以用于线性回归的⼀个原因是：我们假设了观测中包含噪声，其中噪声服从正态分布。噪声正态分布如下式:

$$y=\omega^{T}x+b+\epsilon$$

其中

$$\epsilon \sim \mathcal{N}(0,\sigma^{2})$$

因此,现在我们可以写出给定x观测到的特定y的似然(likelihood)

$$P(y|x)=\frac{1}{\sqrt{2\pi\sigma^{2}}}exp(-\frac{1}{2\sigma^{2}}(y-w^{T}x-b)^2)$$

现在,根据极大似然估计法,参数$\omega$和$b$的最优值是使整个数据集的似然最大的值:

$$P(y|X)=\underset{i=1}{\overset{n}{\prod}}p(y^{(i)}|x^{(i)})$$

根据极大似然估计法选择的估计量称为极⼤似然估计量。虽然使许多指数函数的乘积最⼤化看起来很困难，但是我们可以在不改变目标的前提下，通过最大化似然对数来简化<br>

由于历史原因,优化通常指的是最小化,我们可以改为最小化对数似然,由此得到的数学公式是

$$-logP(y|X)=\overset{n}{\underset{i=1}{\sum}}\frac{1}{2}log(2\pi\sigma^{2})+\frac{1}{2\sigma^{2}}(y^{(i)}-\omega^{T}x^{(i)}-b)^2$$

只需要假设$\sigma$是某个固定常数就可以忽略第一项,因为第一项不依赖于$\omega$和$b$,现在第二项除了$\frac{1}{\sigma^{2}}$外,其余部分与均方差是一样的,而上述式子的解并不依赖于$\sigma$.因此在高斯噪声的假设下,最小化均方误差等价于对线性模型的极大似然估计

<h2>对数几率回归(Logistic)</h2>

Logistic回归是一种常见的处理二分类问题的线性模型.采用$y\in\{0,1\}$以符合logistic的分类习惯

<br>

为了解决连续的线性函数不适合进行分类的问题,引入非线性函数$g:\mathbb{R}^{D}\rightarrow (0,1)$来预测类别标签的后验概率$P(y=1|x)$

$$p(y=1|x)=g(f(x;\omega))$$

其中g通常称为**激活函数(Activation Function)**,其作用是把线性函数的值域从实数区间挤压到(0,1)之间,可以表示概率

在logistics回归中,我们使用Logistic函数作为激活函数,标签$y=1$的后验概率为

$$p(y=1|x)=\sigma (\omega^{T}x)\triangleq \frac{1}{1+exp(-\omega^{T}x)}$$

其中

$$x=[x_1,\dots,x_D,1]^T$$
$$\omega =[\omega_1,\dots,\omega_D,b]^T$$
分别为D+1维的增广特征向量和增广权重向量

标签y=0的后验概率为
$$p$$

<h2>LDA</h2>

