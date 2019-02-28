# =============================================================================
# 深度前馈网络框架
#     多项式(正弦)拟合任务，需要numpy包和matplotlib包；
# =============================================================================

# ------------------ 定义深度前馈网络 -------------------------------------
import numpy as np
import matplotlib.pyplot as plt


class MyDfn:
    __WtInitVar = 0.01  # 初始权值服从标准正态分布，该参数控制方差
    __BsInitAmp = 0.01  # 初始阈值服从均匀分布，该参数控制取值范围
    __miu = 0.02  # 随机梯度下降学习率

    # 网络初始化函数
    def __init__(self, InputLen=1, LayerNum=0, UnitNum=[], ActiveFcs=[]):
        self.LayerNum = LayerNum  # 网络层数(int)
        self.InputLen = InputLen  # 网络输入数据长度(int)
        self.UnitNum = []  # 网络各层的单元数,numpy数组
        self.ActiveFcs = []  # 网络各层的激活函数list,内容为函数指针

        self.WeightMts = []  # 网络各层权值list,内容为numpy矩阵
        self.BiasVc = []  # 网络各层阈值list,内容为numpy矩阵

        # 如果网络层数等于0
        if (self.LayerNum == 0):
            return
        # 每层网络的单元数目
        if (UnitNum.size == LayerNum):
            self.UnitNum = UnitNum
        else:
            print("UnitNum长度和LayerNum不等")
            return
        # 每层网络的激活函数和导数对应的函数指针
        if (len(ActiveFcs) != self.LayerNum):
            print("ActiveFcs维度有误")
            return
        else:
            self.ActiveFcs = ActiveFcs
        # 初始化网络
        self.WeightMts.append(self.__WtInitVar * np.random.randn(UnitNum[0], InputLen))
        self.BiasVc.append(self.__BsInitAmp * np.random.rand(UnitNum[0], 1))
        for idi in range(1, self.LayerNum):
            self.WeightMts.append(self.__WtInitVar * np.random.randn(UnitNum[idi], UnitNum[idi - 1]))
            self.BiasVc.append(self.__BsInitAmp * np.random.rand(UnitNum[idi], 1))

    # 显示网络结构函数
    def PrintNetworkInfo(self):
        print("网络层数=%d" % self.LayerNum)
        if (self.LayerNum >= 1):
            print("第1层：输入数据长度=%d，该层单元数=%d" % (self.InputLen, self.UnitNum[0]))
            for idi in range(1, self.LayerNum):
                print("第%d层：输入数据长度=%d，该层单元数=%d" % (idi + 1, self.UnitNum[idi - 1], self.UnitNum[idi]))

    # 前馈函数(Input为numpy列向量)
    def Forward(self, Input):
        if (Input.shape != (self.InputLen, 1)):
            print("输入数据维度和网络不符")
            return 0.0
        # 第一个元素是网络输入值，后面依次是各层输出值
        self.LyVals = [Input]   # self.LyVals是一个长度为(self.LayerNum+1)的列表,输出计算结果
        self.LyDris = []        # self.LyDris是一个长度为self.LayerNum的列表，每个元素都是对应层输出的导数
        for idi in range(self.LayerNum):
            ZVal = np.dot(self.WeightMts[idi], self.LyVals[idi]) + self.BiasVc[idi]  # 防射
            ValTmp, DriTmp = self.ActiveFcs[idi](ZVal)  # Sigmoid函数进行非线性变换
            self.LyVals.append(ValTmp)
            self.LyDris.append(DriTmp)
        return self.LyVals[self.LayerNum]

    # 反向传播函数(LossDri为numpy列向量)
    def BackPropagation(self, LossDri):
        '''
        :param LossDri: 输出层的导数
        :return: None
        '''
        ErrPost = LossDri * self.LyDris[self.LayerNum - 1]  # 最后的隐藏层的导数
        DeltaErr = [ErrPost]  # 保留子节点的导数
        # 从上到下迭代计算梯度
        for idi in range(self.LayerNum - 2, -1, -1):
            ErrPri = np.dot(self.WeightMts[idi + 1].T, ErrPost) * self.LyDris[idi]
            DeltaErr.append(ErrPri)
            ErrPost = ErrPri
        # 更新参数
        for idi in range(self.LayerNum):
            self.WeightMts[idi] -= self.__miu * np.dot(DeltaErr[self.LayerNum - 1 - idi], self.LyVals[idi].T)
            self.BiasVc[idi] -= self.__miu * DeltaErr[self.LayerNum - 1 - idi]


# ----------------------激活函数必须前一个返回数值，后一个返回导数--------------------
# 激活函数Sigmoid及其导数(第一个numpy向量是函数，第二个是导数)
def Sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y, y * (1 - y)


# 激活函数ReLu及其导数(第一个numpy向量是函数，第二个是导数)
def ReLU(x):
    y = x
    y[x <= 0] = 0
    Dri = np.ones(x.shape)
    Dri[x <= 0] = 0
    return y, Dri


# 线性单元的激活函数及导数(第一个numpy向量是函数，第二个是导数)
def Linear(x):
    return x, np.ones(x.shape)


# MSE均方误差损失函数及其导数(第一个是函数，第二个是导数)
def MSELoss(y, Label):
    diff = y - Label
    return np.dot(diff.T, diff), diff  # 这里的倒数进行了缩放处理
    #return np.dot(diff.T, diff), 2.0 * diff


# ------------------------------------------------------------------------

# -------------------- 进行多项式拟合 -------------------------------------
if __name__=='__main__':


    DatNum = 100  # 回归点数
    # 构造数据
    xdat = np.linspace(-3, 3, DatNum)
    ydat = np.sin(xdat) + 0.01 * np.random.randn(xdat.size)

    # 构建网络
    LyNum = 3  # 网络层数
    UtNum = np.array([20, 12, 1])  # 网络各层的单元数

    # 设计每层激活函数层
    ActFc = []
    for idj in range(LyNum - 1):
        ActFc.append(Sigmoid)
    ActFc.append(Linear)

    # 构造前馈网络
    Net = MyDfn(1, LyNum, UtNum, ActFc)

    # 显示前馈网络结构
    Net.PrintNetworkInfo()

    # 开始训练网络
    IterNum = 3000  # 迭代次数
    for Iter in range(IterNum):
        AveLoss = 0.0
        for DatIdx in range(DatNum):
            InputVal = np.array(xdat[DatIdx]).reshape([1, 1])
            LabelVal = np.array(ydat[DatIdx]).reshape([1, 1])
            OutputVal = Net.Forward(InputVal)
            Loss, LossDri = MSELoss(OutputVal, LabelVal)
            Net.BackPropagation(LossDri)
            AveLoss += Loss[0][0]
        print("第%d次迭代平均损失为%.3f" % (Iter + 1, AveLoss / DatNum))

    # 展示训练成果
    NetOutput = np.zeros(DatNum)
    for DatIdx in range(DatNum):
        InputVal = np.array(xdat[DatIdx]).reshape([1, 1])
        OutputVal = Net.Forward(InputVal)
        NetOutput[DatIdx] = OutputVal[0][0]
    plt.scatter(xdat, ydat)
    plt.plot(xdat, NetOutput, 'r', lw=5)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend(labels=["Net Output", "Train Data"])
    plt.show()

