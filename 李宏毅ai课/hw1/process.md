# 这是hw1的学习过程:covid-19 prediction

## preparation
本套课程将会学习machine learning,集中在deep learning的方法;
观看视频,认知ai两大工作:regression和classification及text,image;
观看视频,了解了完成一个lecture的所需步骤;
1.准备数据
2.选取合适的模型,架构等
3.计算拟合程度(loss) , tensor的梯度计算等
4.优化
5.选择
认识了sigmod,relu等线性模型;

***

## day1

linear layer(线性层) y=Wx+b;
观看了hw1的指导视频;
熟悉colab和kaggle的使用;
本hw使用的是MSE的loss计算(即方差)
do what:
认知了hw1的具体任务,通过找到测试集中的关键feature并修改feat_selection函数,使得score达到medium水平
barrier:
不能很好的关联各部分的函数,在上下文中容易迷糊

***

## day2

进入第二阶段,我尝试加入神经元的数量,虽有成效,但似乎一味加大带来的提升会越来越小,难以达到strong;
而后追加了一层层数和把激活层relu替换成了leakyrelu(对于遇到负数就会梯度消失的优化),效果一般,对model architectures的修改告一段落,转向training loop中的optimizers;
通过阅读链接,得知:
SGD（随机梯度下降）：标准的梯度下降方法，每次更新仅使用一小批数据来估计梯度。它简单但可能需要更多的调整和更长的时间来收敛。
Adam（自适应矩估计）：结合了动量和自适应学习率的特点，通常能更快地收敛。它计算每个参数的自适应学习率，通常在实践中表现更好，但可能更复杂和计算密集。
故打算将SGD替换为Adam:
原:optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9);(momentum用于加速学习过程)
后:optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']);
score不降反增....
尝试加入L2正则化,这个过程我是加入weight_decay来实现的(weight_decay参数直接添加L2正则化，因为在每次更新权重时，它会自动将权重乘以一个略小于1的因子，从而对权重进行惩罚。这相当于在损失函数中添加了一个与权重平方成比例的项，即L2正则化项，有助于防止模型过拟合，使权重保持较小的值。)
没有一点点用....
改用和SGD很像的ASGD(无momentum参数)
不如SGD...
结合几次测试,改回SGD+L2,并在每次激活前添加nn.BatchNorm1d(n)进行归一化
越来越差哈,去掉归一化,加大神经元和层数(128,4).不如(64,3)
改成了(64,4),略有提升.距离双strong还差一些.

***

## day3

重新探究了Adam算法:
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
不停尝试,最后距离public的strong还差0.001几...瓶颈期,pass先