import numpy as np
import scipy as sp
import scipy.linalg as sl

"""
RANSAC算法的主要函数data：输入数据。
model：用于拟合数据的模型。
n：每次迭代中用于拟合模型的样本数量。
k：最大迭代次数。
t：判断数据点是否为内点的阈值。
d：模型被认为是有效的最小内点数量。
debug：是否打印调试信息。
return_all：是否返回所有信息
"""
def ransac(data,model,n,k,t,d,debug = False, return_all = False):
    iterations = 0 #初始化变量，迭代次数
    bestfit = None #最佳拟合模型
    besterr = np.inf  # 设置默认值，最佳误差
    best_inlier_idxs = None #最佳内点索引
    while iterations < k: #只要迭代次数小于最大迭代次数，就继续迭代
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) #random_partition函数用于将数据随机分割为两部分，然后返回两部分的索引，从数据中随机选择n个样本点用于拟合模型，另一部分用于测试模型
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型，现在要用刚才挑出来的maybe_inliers这些数据来训练他，让模型学会这些数据的特点，训练完之后，我们就得到了一个训练好的模型，给它起了个名字叫maybemodel
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小 这句代码是在计算test_points这个测试数据集在maybemodel这个模型下的误差。get_error函数的作用就是计算给定数据在模型下的误差，test_err就存储了测试误差，主要是看模型测试的准不准，预测的错误有多少
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t] #从test_idxs中找出误差小于t的样本点索引，并将这些索引存储在also_idxs中。这些点被认为是内点
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :] #使用also_idxs从数据中获取样本点，并将它们存储在also_inliers中
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # if len(also_inliers > d):
        print('d = ', d)
        ##这段代码是在使用 RANSAC（随机采样一致性）算法来拟合一个模型到一组可能含有噪声的数据。RANSAC 算法通过反复地随机采样数据子集来拟合模型，并尝试找到最优的模型参数，即使数据集中存在大量的噪声或离群点
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接，这行代码将 maybe_inliers 和 also_inliers 两个数组（可能是数据集的不同子集）连接成一个新的数组 betterdata。np.concatenate 是 NumPy 库中的函数，用于连接两个或多个数组。
            bettermodel = model.fit(betterdata)  #使用 fit 方法，将 betterdata 作为输入来训练或拟合模型。bettermodel 是训练后的模型。
            better_errs = model.get_error(betterdata, bettermodel)  #使用 get_error 方法，计算 betterdata 中每个数据点相对于 bettermodel 的误差。better_errs 是一个数组，包含每个数据点的误差
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差，使用 NumPy 的 mean 函数计算 better_errs 数组的平均值，即所有数据点的平均误差。这个平均值可能用于评估模型的质量或用于后续的比较
            if thiserr < besterr:  #这行代码是一个条件判断，它检查当前迭代得到的平均误差（thiserr）是否小于之前的最小误差（besterr）。
                bestfit = bettermodel #如果当前迭代的平均误差更小，那么将当前的模型（bettermodel）设置为最佳模型（bestfit）
                besterr = thiserr #同时，更新最小误差（besterr）为当前的平均误差（thiserr）。
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None: #这行代码检查bestfit（即当前最佳模型）是否为None。如果bestfit是None，表示在设定的迭代次数内没有找到符合输入参数标准的模型
        raise ValueError("did't meet fit acceptance criteria") #如果bestfit是None，则执行raise语句，抛出一个ValueError异常，并附带一条消息"didn't meet fit acceptance criteria"，表示没有找到符合接受标准的拟合模型。
    if return_all: #这个条件语句检查return_all参数是否为True。如果return_all为True，表示需要返回更详细的拟合结果
        return bestfit, {'inliers': best_inlier_idxs} #如果return_all为True，函数将返回两个元素：最佳模型bestfit和一个字典，字典中包含用于生成该模型的内点索引best_inlier_idxs
    else:
        return bestfit #当return_all为True时，函数会返回更详细的拟合结果；否则，只返回最佳模型。


def random_partition(n, n_data): #random_partition 函数的作用是从给定的数据集中随机选择指定数量的行，并返回这些行的索引，同时返回其他所有行的索引。这个函数常用于需要从大数据集中随机选择部分数据来进行模型拟合或测试的情况
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data) #获取n_data下标索引，np.arange(n_data) 生成从0到n_data-1的整数序列，用于表示数据集的所有索引
    np.random.shuffle(all_idxs) #打乱下标索引，np.random.shuffle(all_idxs) 打乱all_idxs中的索引，使得每次调用random_partition函数时，返回的随机行索引都会不同
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:] #idxs1 = all_idxs[:n] 和 idxs2 = all_idxs[n:] 分别提取前n个索引和剩余索引，作为返回的两个值。函数返回两个列表idxs1和idxs2，分别表示随机选择的行索引和其他行的索引。这可以用于从原始数据中提取相应的子集进行后续处理。

    return idxs1, idxs2
#这段代码的核心原理是利用NumPy库中的arange函数生成索引序列，然后使用random.shuffle函数对索引进行随机排序，最后根据所需的行数来分割索引列表，返回两部分索引，一部分是随机选择的行索引，另一部分是剩余的行索引。这种随机分割的方法常用于机器学习中从大量数据中随机选择一部分数据进行模型训练或验证

class LinearLeastSquareModel: #class LinearLeastSquareModel:是定义一个名为LinearLeastSquareModel的类。类是面向对象编程中的核心概念，它定义了对象的属性和方法。在这个例子中，LinearLeastSquareModel类被设计为用于通过最小二乘法求解线性模型的参数
    #最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug = False): #def __init__(self, input_columns, output_columns, debug = False):是LinearLeastSquareModel类的初始化方法（也称为构造函数）。当你创建一个新的LinearLeastSquareModel类的实例时，这个方法会被自动调用,__init__是一个特殊方法，在创建类的新实例时自动调用
        self.input_columns = input_columns  #self：这是一个对类实例本身的引用，它允许你在类的内部访问该实例的属性和方法 input_columns：这个参数通常是一个表示输入特征（自变量）的列表或数组 utput_columns：这个参数通常是一个表示输出（因变量）的列表或数组 debug：这是一个可选参数，它有一个默认值False。当你创建一个新的LinearLeastSquareModel实例时，如果你没有提供debug参数，那么它的值就是False。这个参数通常用于控制是否开启调试模式
        self.output_columns = output_columns #self.input_columns = input_columns：这行代码将传入的input_columns参数值赋给实例变量input_columns。这样，类的实例就可以通过self.input_columns来访问这些输入列的信息 self.output_columns = output_columns：类似地，这行代码将output_columns参数值赋给实例变量output_columns，使得实例可以通过self.output_columns访问输出列的信息
        self.debug = debug #将方法参数中的debug（具有默认值False）赋值给实例变量debug。这样，就可以通过self.debug来控制是否开启调试模式

    def fit(self, data): #这行代码定义了一个名为fit的方法，它是LinearLeastSquareModel类的一个实例方法。fit方法通常用于训练模型，通过调整模型参数来最小化损失函数。在这个方法中，data参数表示要用于训练的数据
		#np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi 这行代码使用np.vstack函数将data中的特定列垂直堆叠成一个新的数组A。[data[:,i] for i in self.input_columns]是一个列表推导式，它选择data中对应于self.input_columns（即输入列索引）的所有列，并对这些列进行垂直堆叠。.T用于转置矩阵，将行变为列，将列变为行
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi 这行代码与上面的代码类似，但是它是将data中对应于self.output_columns（即输出列索引）的所有列垂直堆叠成一个新的数组B
        x, resids, rank, s = sl.lstsq(A, B) #residues:残差和 这行代码调用sl.lstsq函数，该函数可能是用于计算最小二乘解的函数。A和B分别作为输入和输出矩阵传递给函数。函数的返回值被解包为四个变量：x（最小平方和向量）、resids（残差和）、rank（秩）和s（可能是一个表示解的特性的值
        return x #返回最小平方和向量

    def get_error(self, data, model):
        k, b = model  # 假设model是一个包含斜率和截距的元组
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 计算预测的Y值，注意这里我们使用了斜率k和截距b
        B_fit = A.dot(k) + b  # 如果k是向量，则需要使用np.dot(A, k.reshape(-1, 1))
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算每个点的平方误差
        return err_per_point

    """
    线性回归模型： 线性回归模型通常表示为 y = kx + b，其中 y 是因变量（预测值或响应变量），x 是自变量（解释变量或特征），k 是斜率（或称为系数），b 是截距。在多元线性回归中，x 可以是一个向量，k 是一个与 x 中每个元素相对应的系数向量。

矩阵乘法（在修改后的示例中未直接使用，但在原始代码中可能误导性地使用了）： 如果 A 是一个矩阵，且 k 是一个与 A 列数相匹配的向量（在多元线性回归的上下文中），则 A.dot(k) 会计算矩阵 A 和向量 k 的点积，结果是一个向量，其每个元素是 A 的行与 k 的对应元素乘积的和。然而，在单变量线性回归中，我们通常不将斜率 k 视为向量进行矩阵乘法，而是直接将其与自变量相乘。

预测值计算： 对于给定的自变量值 x（或矩阵 A 中的行），预测值 y_pred（或 B_fit）通过线性方程 y_pred = kx + b（或对于矩阵形式，如果 k 是向量则为 y_pred = A.dot(k) + b，但注意 b 需要是一个与 y_pred 形状相同的向量或标量广播到该形状）计算得出。

误差计算（平方误差）： 对于每个观测点，实际值 y 和预测值 y_pred 之间的差异称为误差。平方误差是误差的平方，用于避免正负误差相互抵消。平方误差的总和（或平均）常用于评估模型的性能。在这段代码中，通过 np.sum((B - B_fit) ** 2, axis=1) 计算了每个观测点的平方误差，其中 B 是实际值，B_fit 是预测值，axis=1 表示沿着行的方向（即对每个观测点）进行求和
    """


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi1

    # 这是一个条件判断，但实际上总是为真（因为条件是1，非零即为真），所以下面的代码块会执行。
    if 1:
        # 添加"局外点"
        # 定义要添加的局外点数量
        n_outliers = 100
        # 获取A_noisy数据集的索引，假设A_noisy是一个形状为(500, n_inputs)的数组，这里生成0到499的索引
        all_idxs = np.arange(A_noisy.shape[0])
        # 将索引数组打乱，以便随机选择局外点
        np.random.shuffle(all_idxs)
        # 选择前n_outliers个索引作为局外点的索引
        outlier_idxs = all_idxs[:n_outliers]
        # 在A_noisy的局外点索引处添加噪声，生成一个形状为(n_outliers, n_inputs)的随机数组，并乘以20
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        # 在B_noisy的局外点索引处也添加噪声，生成一个形状为(n_outliers, n_outputs)的正态分布随机数组，并乘以50
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

        # setup model
        # 将A_noisy和B_noisy水平堆叠成一个新的数组all_data，其中每行包含对应的输入和输出数据
        all_data = np.hstack((A_noisy, B_noisy))
        # 定义输入数据的列索引，这里假设输入数据位于all_data的前n_inputs列
        input_columns = range(n_inputs)
        # 定义输出数据的列索引，这里假设输出数据紧跟在输入数据之后，即索引从n_inputs到n_inputs+n_outputs-1
        output_columns = [n_inputs + i for i in range(n_outputs)]
        # 设置调试标志为False，这里可能用于控制LinearLeastSquareModel类中的调试输出
        debug = False
        # 实例化LinearLeastSquareModel类，传入输入和输出列索引以及调试标志
        model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)

        # 使用线性最小二乘法拟合数据
        # sp.linalg.lstsq是SciPy库中的线性最小二乘解算器，这里用它来拟合all_data中的输入和输出数据
        # all_data[:,input_columns]是输入数据，all_data[:,output_columns]是输出数据
        linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

        # run RANSAC 算法
        # 使用RANSAC算法对数据进行鲁棒拟合，以减少局外点的影响
        # ransac函数可能是自定义的或来自某个库，它接受数据集、模型、最小样本数、最大迭代次数、阈值、残差等参数
        # 这里假设它返回拟合结果ransac_fit和用于拟合的内点数据ransac_data
        # 注意：这里的参数50, 1000, 7e3, 300分别代表最小样本数、最大迭代次数、阈值和残差，但具体含义可能因ransac函数的实现而异
        ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:  # 这实际上是一个总是为真的条件判断，因此下面的代码块会执行。
        import pylab  # 导入pylab模块，它通常是一个集成了NumPy和matplotlib的模块，用于绘图。

        # 对A_exact的第一列进行排序，并获取排序后的索引。
        sort_idxs = np.argsort(A_exact[:, 0])
        # 使用排序后的索引对A_exact进行索引，得到按第一列排序后的数组。
        A_col0_sorted = A_exact[sort_idxs]

        # 下面的if-else结构看起来有些不寻常，因为if条件总是为真（1），所以else分支永远不会被执行。
        # 但如果这里是为了演示目的而保留的，我们可以忽略它，并假设else分支是为了说明另一种可能的绘图方式。
        if 1:  # 总是为真的条件
            # 绘制所有噪声数据的散点图，用黑色点表示。
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            # 绘制RANSAC算法识别出的内点数据，用蓝色'x'表示。
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:  # 这部分实际上永远不会被执行
        # 如果else分支被执行，它会绘制非局外点和局外点的数据，但这里不会。
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
        # 绘制RANSAC拟合曲线。首先，使用排序后的A_col0_sorted数组和RANSAC拟合参数ransac_fit计算拟合值。
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        # 绘制精确系统（即没有噪声和局外点的系统）的拟合曲线。
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        # 绘制线性最小二乘拟合曲线。
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')

        # 添加图例，显示每条曲线的标签。
        pylab.legend()
        # 显示图形。
        pylab.show()

    if __name__ == "__main__":
        test()  # 调用test函数，该函数应该在其他地方定义，并负责设置上述所有变量。