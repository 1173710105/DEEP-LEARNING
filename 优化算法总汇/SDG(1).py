import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

'''
随机梯度下降算法每次只随机选择一个样本来更新模型参数，因此每次的学习是非常快速的，并且可以进行在线更新
'''

def createDataSet():
    # 构造训练数据
    x = np.arange(-10, 10, 0.5)  # 根据start与stop指定的范围以及step设定的步长, 获去数据集, 返回一个列表
    m = len(x)  # 训练数据点数目
    target_data = x * 2 + 5 + np.random.randn(m)  # 随机设置随即设置x对应的y值
    return x, target_data

def SGD(x, target_data, loop_max=1000,epsilon=1e-2):
    '''
    :param x: 训练数据点X轴坐标
    :param target_data: 训练数据点Y轴坐标
    :param input_data:
    :param loopmax: 终止条件,最大迭代次数(防止死循环)
    :param epsilon: 终止条件,目标函数与拟合函数的距离当的距离小于epsilon时, 退出
    :return: theta 训练结束之后的参数
    '''

    m = len(x)  # 训练数据点数目
    x0 = np.full(m, 1.0)  # 获取一个长度为m, 每个元素都是1.0 的列表
    # T 表示转秩矩阵, 不加T 第一行元素为依次为x0,第二行元素依次为x, 得到2*m矩阵
    input_data = np.vstack([x0, x]).T  # 构造矩阵, 第一列元素为依次为x0,第二列元素依次为x,得到m*2矩阵,方便做矩阵乘法

    # 初始化权值
    np.random.seed(0)  #设置随机数种子
    theta = np.random.randn(input_data.shape[1])  # 初始化theta,, 生成一个列表,维度与输入空间一致

    alpha = 0.01  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
    diff = 0.0  # 记录梯度
    count = 0  # 循环次数
    error = 0  # 记录误差


    while count < loop_max:
        count += 1
        gradient = np.zeros(2)
        # 随机梯度下降的权值是通过考查眸一小批样例来更新的
        index = random.sample(range(m), int(np.ceil(m * 0.2)))
        # 遍历训练数据集，不断更新权值
        for i in index:
            # 以下数求损失函数的梯度
            diff = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
            gradient += diff
        gradient = gradient/len(index)
        theta = theta - alpha * gradient  # 注意步长alpha的取值,过大会导致振荡,过小收敛速度太慢
        error = calMSE(x, target_data, theta)  # 计算均方误差
        if error < epsilon:
            break
        print('loop count = %d' % count, '\tw:', theta, 'error=', error)
    return theta

    return theta
def draw(x,target_data,theta):
    # check with scipy linear regression
    # intercept 回归曲线的截距
    # slope 回归曲线的斜率
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
    print('intercept = %s slope = %s' % (intercept, slope))
    plt.plot(x, target_data, 'g.')
    plt.plot(x, x * theta[1] + theta[0], 'r')
    plt.show()

def calMSE(x,target_data,theta):
    MSE = 0
    for i in range(len(x)):
        temp = x[i]*theta[1]+theta[0] - target_data[i]
        MSE += temp*temp
    return MSE/len(x)


if __name__=='__main__':
    x, target_data= createDataSet()
    theta = SGD(x,target_data)
    draw(x,target_data,theta)
