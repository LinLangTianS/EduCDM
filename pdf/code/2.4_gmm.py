import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['font.sans-serif'] = ['Kai']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成高斯混合数据
n_samples = 1000

# 定义三个高斯分量的参数
mu1, mu2, mu3 = -2.0, 0.0, 4.0
sigma1, sigma2, sigma3 = 0.5, 0.8, 1.5

# 混合系数（权重）
true_alpha = [0.3, 0.4, 0.3]

# 生成数据
n1 = int(true_alpha[0] * n_samples)
n2 = int(true_alpha[1] * n_samples)
n3 = n_samples - n1 - n2

X1 = np.random.normal(mu1, sigma1, n1)
X2 = np.random.normal(mu2, sigma2, n2)
X3 = np.random.normal(mu3, sigma3, n3)
X = np.concatenate([X1, X2, X3])

# 打乱数据顺序
np.random.shuffle(X)

# 记录真实的类别标签
true_labels = np.concatenate([np.zeros(n1), np.ones(n2), 2*np.ones(n3)])
order = np.argsort(X)
true_labels = true_labels[order]
X = X[order]

# 高斯分布概率密度函数
def gaussian_pdf(x, mu, sigma):
    """
    计算高斯分布的概率密度
    """
    return norm.pdf(x, loc=mu, scale=sigma)

# EM算法实现
def EM_GMM(X, K, max_iter=100, tol=1e-6):
    """
    使用EM算法估计一维高斯混合模型的参数
    
    Args:
        X: 数据点 [n_samples]
        K: 高斯分量数量
        max_iter: 最大迭代次数
        tol: 收敛阈值
        
    Returns:
        alpha: 混合系数
        mu: 均值列表
        sigma: 标准差列表
        history: 迭代历史记录
    """
    n = len(X)  # 样本数
    
    # 随机初始化参数
    alpha = np.ones(K) / K  # 均匀分布的初始混合系数
    
    # 根据数据范围设置初始均值
    min_x = np.min(X)
    max_x = np.max(X)
    # 在数据范围内均匀分布初始均值
    mu = np.linspace(min_x, max_x, K)
    
    # 初始标准差 - 使用数据的标准差
    sigma = np.ones(K) * np.std(X) / np.sqrt(K)
    
    # 记录历史参数
    history = {
        'alpha': [alpha.copy()],
        'mu': [mu.copy()],
        'sigma': [sigma.copy()],
        'log_likelihood': []
    }
    
    for iteration in range(max_iter):
        # === E步 ===
        gamma = np.zeros((n, K))
        
        for k in range(K):
            gamma[:, k] = alpha[k] * gaussian_pdf(X, mu[k], sigma[k])
        
        row_sums = gamma.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1e-10
        gamma = gamma / row_sums
        
        # === M步 ===
        N_k = gamma.sum(axis=0)
        alpha_new = N_k / n
        
        mu_new = np.zeros(K)
        for k in range(K):
            mu_new[k] = np.sum(gamma[:, k] * X) / N_k[k]
        
        sigma_new = np.zeros(K)
        for k in range(K):
            diff = X - mu_new[k]
            sigma_new[k] = np.sqrt(np.sum(gamma[:, k] * diff**2) / N_k[k])
            
            sigma_new[k] = max(sigma_new[k], 1e-3)
        
        log_likelihood = 0
        for i in range(n):
            s = 0
            for k in range(K):
                s += alpha[k] * gaussian_pdf(X[i], mu[k], sigma[k])
            log_likelihood += np.log(s)
        
        # 保存历史
        history['alpha'].append(alpha_new.copy())
        history['mu'].append(mu_new.copy())
        history['sigma'].append(sigma_new.copy())
        history['log_likelihood'].append(log_likelihood)
        
        # 检查收敛
        if iteration > 0:
            ll_diff = abs(log_likelihood - history['log_likelihood'][-2])
            if ll_diff < tol:
                print(f"收敛于第 {iteration+1} 次迭代，对数似然差: {ll_diff:.8f}")
                break
        
        # 更新参数
        alpha = alpha_new
        mu = mu_new
        sigma = sigma_new
        
        if (iteration + 1) % 10 == 0:
            print(f"迭代 {iteration+1} - 对数似然: {log_likelihood:.4f}")
    
    if iteration == max_iter - 1:
        print(f"达到最大迭代次数 {max_iter}")
    
    return alpha, mu, sigma, history

# 可视化函数
def plot_gmm(X, alpha, mu, sigma, history):
    """
    可视化高斯混合模型
    """
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    # 计算数据点的后验概率
    n = len(X)
    gamma = np.zeros((n, len(mu)))
    for k in range(len(mu)):
        gamma[:, k] = alpha[k] * gaussian_pdf(X, mu[k], sigma[k])
    
    gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]
    labels = np.argmax(gamma, axis=1)
    
    # 图1：绘制直方图，不同颜色表示不同分量
    plt.figure(figsize=(10, 6))
    plt.hist(X, bins=30, density=True, alpha=0.5, color='gray', 
             edgecolor='black', label='数据直方图')
    
    # 各分量的直方图
    for k in range(len(mu)):
        idx = labels == k
        plt.hist(X[idx], bins=20, density=True, alpha=0.3,
                 color=colors[k], label=f'分量 {k+1}')
    
    plt.xlabel('数值')
    plt.ylabel('频率密度')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gmm_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图2：概率密度函数
    plt.figure(figsize=(10, 6))
    x_grid = np.linspace(np.min(X) - 2*np.std(X), np.max(X) + 2*np.std(X), 1000)
    
    # 绘制各分量密度
    for k in range(len(mu)):
        density = gaussian_pdf(x_grid, mu[k], sigma[k])
        plt.plot(x_grid, alpha[k] * density, color=colors[k], linestyle='-',
                label=f'分量{k+1}: α={alpha[k]:.2f}, μ={mu[k]:.2f}, σ={sigma[k]:.2f}')
        plt.fill_between(x_grid, alpha[k] * density, alpha=0.2, color=colors[k])
    
    # 绘制混合分布
    mixture_density = np.zeros_like(x_grid)
    for k in range(len(mu)):
        mixture_density += alpha[k] * gaussian_pdf(x_grid, mu[k], sigma[k])
    
    plt.plot(x_grid, mixture_density, color='black', linestyle='-', linewidth=2,
            label='混合分布')
    
    # 添加数据点分布
    plt.hist(X, bins=30, density=True, alpha=0.3, color='gray',
             edgecolor='black')
    
    plt.xlabel('数值')
    plt.ylabel('概率密度')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gmm_density.png', dpi=300, bbox_inches='tight')
    plt.show()

alpha_est, mu_est, sigma_est, history = EM_GMM(X, K=3, max_iter=100, tol=1e-4)
print("\n估计的混合系数:", alpha_est)
print("\n估计的均值:", mu_est)
print("\n估计的标准差:", sigma_est)

print("\n真实参数:")
print("真实混合系数:", true_alpha)
print("真实均值:", [mu1, mu2, mu3])
print("真实标准差:", [sigma1, sigma2, sigma3])

# 可视化结果
plot_gmm(X, alpha_est, mu_est, sigma_est, history)