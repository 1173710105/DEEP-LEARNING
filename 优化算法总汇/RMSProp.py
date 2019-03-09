import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy import stats


"""
    RMSProp在AdaGrad基础上修改,改变梯度累积为指数加权的移动平均
    RMSProp使用指数衰减平均一对其遥远过去的历史,使其能够在找到凸碗结构后能快速收敛
"""


def createDataSet():
    # 构造训练数据
    x = np.arange(-10, 10, 0.5)  # 根据start与stop指定的范围以及step设定的步长, 获去数据集, 返回一个列表
    m = len(x)  # 训练数据点数目
    target_data = 3 * x + 8 + np.random.randn(m) # 随机设置随即设置x对应的y值
    return x, target_data

def RMSPROP(x, target_data, loop_max=1000,epsilon=1):

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
    theta = np.random.randn(input_data.shape[1])  # 初始化theta,生成一个列表,第一个元素是w(维度与输入空间一致), 第二个元素是b,

    eps = 0.1  # 步长
    diff = 0. # 记录梯度
    count = 0  # 循环次数
    alpha= 0.9  # 衰减力度,可以用来调节,该值越大那么之前的梯度对现在方向的影响也越大
    error = 0 # 记录误差
    r = np.zeros(2)  # 梯度累计量
    p = 0.1  # 梯度衰减速率
    while count < loop_max:
        count += 1
        gradient = np.zeros(2)
        index = random.sample(range(m), int(np.ceil(m * 0.2)))
        for i in index:
            diff = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
            gradient = gradient + diff
        gradient = gradient/len(index)
        r = p*r + (1-p)*(gradient*gradient)  # 内径平方梯度
        for i in range(0, theta.size):
            theta[i] = theta[i] + (-eps/math.sqrt(1e-6+r[i]))*gradient[i]  # 逐元素更新theta
        error = calMSE(x, target_data, theta)  # 计算均方误差
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
    theta = RMSPROP(x,target_data)
    draw(x,target_data,theta)

