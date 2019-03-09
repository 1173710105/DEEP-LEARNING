import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy import stats


"""
    在Adam中,动量首先并入了梯度一阶矩的估计,将动量加入RMSProp最直观的方法是将最直观的方法是
    将动量应用应用缩放之后梯度;其次,Adam包括偏置修正,修正从原点初始化的一阶矩和二阶矩的估计
"""


def createDataSet():
    # 构造训练数据
    x = np.arange(-10, 10, 0.5)  # 根据start与stop指定的范围以及step设定的步长, 获去数据集, 返回一个列表
    m = len(x)  # 训练数据点数目
    target_data = 3 * x + 8 + np.random.randn(m) # 随机设置随即设置x对应的y值
    return x, target_data

def Adam(x, target_data, loop_max=10000,epsilon=1):

    '''
    :param x: 训练数据点X轴坐标
    :param target_data: 训练数据点Y轴坐标
    :param input_data:
    :param loopmax: 终止条件,最大迭代次数(防止死循环)
    :param epsilon: 终止条件,目标函数与拟合函数的距离当的距离小于epsilon时,或者更新梯度之后变化小于epsilon时,退出
    :return: theta 训练结束之后的参数
    '''

    m = len(x)  # 训练数据点数目
    x0 = np.full(m, 1.0)  # 获取一个长度为m, 每个元素都是1.0 的列表
    # T 表示转秩矩阵, 不加T 第一行元素为依次为x0,第二行元素依次为x, 得到2*m矩阵
    input_data = np.vstack([x0, x]).T  # 构造矩阵, 第一列元素为依次为x0,第二列元素依次为x,得到m*2矩阵,方便做矩阵乘法

    # 初始化权值
    np.random.seed(0)  #
    theta = np.random.randn(input_data.shape[1])  # 初始化theta,, 生成一个列表,第一个元素是w(维度与输入空间一致), 第二个元素是b,
    v = np.zeros(2)  # 更新的速度参数

    eps = 0.1  # 步长
    diff = 0. # 记录梯度
    count = 0  # 循环次数
    alpha= 0.9  # 衰减力度,可以用来调节,该值越大那么之前的梯度对现在方向的影响也越大
    error = 0 # 记录误差
    s = np.zeros(2)  # 梯度累计量,更新一阶矩估计
    r = np.zeros(2)  # 梯度累计量,更新二阶矩估计
    p1 = 0.1  # 梯度衰减速率
    p2 = 0.1  # 梯度衰减速率
    while count < loop_max:
        count += 1
        gradient = np.zeros(2)
        index = random.sample(range(m), int(np.ceil(m * 0.2)))
        for i in index:
            diff = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
            gradient = gradient + diff
        gradient = gradient/len(index)  # 计算平局梯度
        s = p1*s + (1 - p1) * gradient  # 更新一阶矩估计
        r = p2*r + (1 - p2) * (gradient*gradient)  # 更新二阶矩估计
        s = s / (1 - math.pow(p1, count))  #修正一阶矩偏差
        r = r / (1 - math.pow(p2, count))  #修正二阶矩偏差
        for i in range(0, theta.size):
            theta[i] = theta[i] - eps*s[i]/(math.sqrt(r[i])+1e-6)  # 逐个元素更新
        error = calMSE(x,target_data,theta)  # 计算均方误差
        if error < epsilon:
            break
        print('loop count = %d' % count, '\tw:', theta, 'error=',error)
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
    theta = Adam(x,target_data)
    draw(x,target_data,theta)
