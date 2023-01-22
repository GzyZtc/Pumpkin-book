<h1>第三章 线性模型</h1>
<h2>线性回归</h2>

给定数据集$D = \{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$,其中$x_i=(x_{i1};_{i2};...;x_{id}),y_i \in \mathbb{R}$,线性回归试图学得

$$f(x+i)=\omega x_i + b,使得f(xi)\simeq y_i$$

其中$\omega,b$使得均方误差最小化

$$(\omega ^{*},b^{*})=\underset{(\omega ,b)}\argmin \overset{m}{\underset{i=1}{\sum}}(f(x_i)-y_i)^2=\underset{(\omega ,b)}\argmin \overset{m}{\underset{i=1}{\sum}}(y_i-\omega x_i-y_i)^2$$




<h2>对数几率回归
<h2>LDA
