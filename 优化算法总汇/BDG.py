import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

'''
优点：全局最优解；易于并行实现；总体迭代次数不多
缺点：当样本数目很多时，训练过程会很慢，每次迭代需要耗费大量的时间
'''

def createDataSet():
    # 构造训练数据
    x = np.arange(-10, 10, 0.5)  # 根据start与stop指定的范围以及step设定的步长, 获去数据集, 返回一个列表
    m = len(x)  # 训练数据点数目
    target_data = x * 2 + 5 + np.random.randn(m)  # 随机设置随即设置x对应的y值
    return x, target_data


def BGD(x, target_data, loop_max=1000,epsilon=1e-2):
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
    np.random.seed(0)  #
    theta = np.random.randn(input_data.shape[1])  # 初始化theta,, 生成一个列表,第一个元素是w(维度与输入空间一致), 第二个元素是b,

    alpha = 0.01  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
    diff = 0.  # 记录梯度
    count = 0  # 循环次数
    error = 0 # 用于记录均方误差
    while count < loop_max:
        count += 1

        # 标准梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
        # 在标准梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算

        sum_m = np.zeros(2)
        for i in range(m):
            # 以下数求损失函数的梯度
            diff = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
            # 可以在迭代theta的时候乘以步长alpha, 也可以在梯度求和的时候乘以步长alpha
            sum_m = sum_m + diff  # 当alpha取值过大时,sum_m会在迭代过程中会溢出
        sum_m = sum_m/m
        if np.linalg.norm(sum_m)<epsilon:  # 设置阀值, 如果梯度过小, 退出
            break
        theta = theta - alpha * sum_m  # 注意步长alpha的取值,过大会导致振荡,过小收敛速度太慢
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
    theta = BGD(x,target_data)
    draw(x,target_data,theta)
