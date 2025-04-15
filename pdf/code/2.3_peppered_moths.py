import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Kai']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def generate_data(n, pc=0.5, pi=0.3, pt=0.2):
    """生成模拟观测数据
    
    Args:
        n (int): 数据样本数量
        pc (float): C等位基因的频率
        pi (float): I等位基因的频率
        pt (float): T等位基因的频率
        
    Returns:
        list: 观测数据列表，元素为 'C', 'I' 或 'T'
    """
    # 计算各表型概率
    prob_c = pc**2 + 2*pc*(pi + pt)  # CC, CI, CT
    prob_i = pi**2 + 2*pi*pt         # II, IT
    prob_t = pt**2                   # TT
    
    # 生成数据
    data = []
    for _ in range(n):
        r = np.random.rand()
        if r < prob_c:
            data.append('C')
        elif r < prob_c + prob_i:
            data.append('I')
        else:
            data.append('T')
    return data

def em(data, p0=None, max_iter=100, tol=1e-6, verbose=True):
    """使用EM算法估计等位基因频率
    
    Args:
        data (list): 观测数据列表
        p0 (list): 初始参数 [pc, pi, pt]
        max_iter (int): 最大迭代次数
        tol (float): 收敛阈值
        verbose (bool): 是否打印迭代信息
        
    Returns:
        tuple: (估计的频率 [pc, pi, pt], 迭代历史)
    """
    # 参数初始化
    if p0 is None:
        p = np.random.dirichlet([1, 1, 1])
    else:
        p = np.array(p0) / sum(p0)
    
    history = {'pc': [p[0]], 'pi': [p[1]], 'pt': [p[2]]}
    for it in range(max_iter):
        # E步：计算期望计数
        count_c, count_i, count_t = 0.0, 0.0, 0.0
        
        for pheno in data:
            pc, pi, pt = p
            if pheno == 'C':
                denom = pc**2 + 2*pc*(pi + pt)
                if denom <= 1e-8:
                    continue
                # 各基因型概率
                prob_cc = (pc**2) / denom
                prob_ci = (2*pc*pi) / denom
                prob_ct = (2*pc*pt) / denom
                # 累加期望计数
                count_c += 2*prob_cc + prob_ci + prob_ct
                count_i += prob_ci
                count_t += prob_ct
                
            elif pheno == 'I':
                denom = pi**2 + 2*pi*pt
                if denom <= 1e-8:
                    continue
                # 各基因型概率
                prob_ii = (pi**2) / denom
                prob_it = (2*pi*pt) / denom
                # 累加期望计数
                count_i += 2*prob_ii + prob_it
                count_t += prob_it
                
            else:  # T表型
                count_t += 2
        
        # M步：更新参数
        total = count_c + count_i + count_t
        if total == 0:
            new_p = np.array([0, 0, 0])
        else:
            new_p = np.array([count_c, count_i, count_t]) / total
        
        # 保存本次迭代结果
        history['pc'].append(new_p[0])
        history['pi'].append(new_p[1])
        history['pt'].append(new_p[2])
        
        # 检查收敛
        delta = np.linalg.norm(new_p - p)
        if verbose:
            print(f"Iteration {it+1}: C={new_p[0]:.4f}, I={new_p[1]:.4f}, T={new_p[2]:.4f}, delta={delta:.6f}")
        if delta < tol:
            break
        p = new_p
    
    return p, history

def plot_convergence(history, true_values=None):
    """绘制迭代收敛过程
    
    Args:
        history (dict): 保存迭代历史的字典
        true_values (list): 真实参数值 [pc, pi, pt]
    """
    iterations = range(len(history['pc']))
    
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, history['pc'], 'r-', label='C基因频率')
    plt.plot(iterations, history['pi'], 'g-', label='I基因频率')
    plt.plot(iterations, history['pt'], 'b-', label='T基因频率')
    
    if true_values is not None:
        plt.axhline(y=true_values[0], color='r', linestyle='--', alpha=0.5, label='真实C频率')
        plt.axhline(y=true_values[1], color='g', linestyle='--', alpha=0.5, label='真实I频率')
        plt.axhline(y=true_values[2], color='b', linestyle='--', alpha=0.5, label='真实T频率')
    
    plt.xlabel('迭代次数')
    plt.ylabel('基因频率')
    # plt.title('EM算法估计胡椒蛾基因频率的收敛过程')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('peppered_moths_convergence.png', dpi=300)
    plt.show()

# 生成模拟数据并估计参数
np.random.seed(42)
true_params = [0.5, 0.3, 0.2]  # 真实的基因频率
data = generate_data(10000, *true_params)

# 随机初始化参数
initial_params = np.random.dirichlet([1, 1, 1])
print(f"初始参数: C={initial_params[0]:.4f}, I={initial_params[1]:.4f}, T={initial_params[2]:.4f}")

# 运行EM算法
estimated_params, history = em(data, initial_params, max_iter=50, tol=1e-6)

print("\nFinal Estimates:")
print(f"C: {estimated_params[0]:.4f}, I: {estimated_params[1]:.4f}, T: {estimated_params[2]:.4f}")
print(f"真实值: C: {true_params[0]:.4f}, I: {true_params[1]:.4f}, T: {true_params[2]:.4f}")

# 绘制收敛过程
plot_convergence(history, true_params)

# 计算并显示误差
error = np.linalg.norm(estimated_params - true_params)
print(f"\n参数估计误差: {error:.6f}")