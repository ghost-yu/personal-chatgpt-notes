# 原notebook Python中文注释阅读版：bradley_terry_modeling
# 所有原代码保留；绘图使用matplotlib，补充BFGS需要真实SciPy。

# %% 原单元 0 · code
# 导入 NumPy 与 SciPy 的 optimize.minimize，用数组组织比赛数据，再估计队伍或品牌的强度。运行环境见复现说明。
# 导入后续计算使用的模块或函数。
import numpy as np
# 导入后续计算使用的模块或函数。
from scipy.optimize import minimize


# %% 原单元 1 · markdown
# 这一格列出 Bradley–Terry 的参考链接。补充例子为每支队伍或品牌直接估计一个分数；神经奖励模型则从输入文本计算分数。
# - https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
# - https://chatgpt.com/c/672eeaa4-4b84-800e-832b-87dad7fa7f66

# %% 原单元 2 · code
# w[i,j] 记录 i 胜 j 的次数，n=w+w.T 记录总比赛数。固定 theta_A=0，优化另外3个对数强度；每个 i<j 只计一次。BFGS 用数值梯度最小化负对数似然。外层最多调用10次 minimize，打印的次数按外层循环计数；收敛分支在写回 result.x 前结束循环。
# 导入后续计算使用的模块或函数。
import numpy as np
# 导入后续计算使用的模块或函数。
from scipy.optimize import minimize

# 胜利次数矩阵
w = np.array([
    [0, 2, 0, 1],
    [3, 0, 5, 0],
    [0, 3, 0, 1],
    [4, 0, 3, 0]
])

# 总比赛次数矩阵
n = w + w.T

# 定义负对数似然，供 SciPy 数值优化。
def neg_log_likelihood(theta):
    # 把被固定的基准log强度0拼回完整向量。
    theta_full = np.concatenate(([0], theta))
    # log强度指数化为正强度；此原实现大输入可能overflow。
    exp_theta = np.exp(theta_full)
    L = 0
    # 遍历对象下标i；后面j从i+1起避免重复一对。
    for i in range(len(w)):
        # 只枚举无序对i<j。
        for j in range(i+1, len(w)):
            # 没有交手数据的对象对不贡献似然。
            if n[i, j] > 0:
                L += (
                    # i获胜次数乘i的log强度。
                    w[i, j] * theta_full[i] +
                    # j获胜次数乘j的log强度。
                    w[j, i] * theta_full[j] -
                    # 减去总次数乘log(exp(theta_i)+exp(theta_j))。
                    n[i, j] * np.log(exp_theta[i] + exp_theta[j])
                )
    # 取负号供最小化器使用；最大似然变为最小负似然。
    return -L

# 设置初始猜测和收敛条件
# 三维自由log强度初始化为0，基准A另固定。
theta_opt = np.zeros(3)
tolerance = 1e-6
max_iterations = 10

# 外层循环重复调用 BFGS，iteration 统计调用次数。
for iteration in range(max_iterations):
    # 真实BFGS优化，未传jac时使用数值差分。
    result = minimize(
        neg_log_likelihood,
        theta_opt,
        method='BFGS'
    )

    # 检查优化器是否成功，不静默接受失败结果。
    if not result.success:
        # 失败时中断此格并保留消息。
        raise Exception('Optimization failed: ' + result.message)

    # 检查参数变化是否小于阈值
    # 比较本次与上次自由参数逐项差，阈值tolerance。
    if np.all(np.abs(theta_opt - result.x) < tolerance):
        print(f"Optimization converged after {iteration + 1} iterations.")
        break

    # 记录本次最优自由参数；break分支在此行之前。
    theta_opt = result.x

# 把被固定的基准log强度0拼回完整向量。
theta_full = np.concatenate(([0], theta_opt))
# 正强度exp(theta)，以A=1为基准。
p = np.exp(theta_full)
# 强度除以总和，得到统一尺度；指定两队的胜率另用 pi_i/(pi_i+pi_j) 计算。
p_normalized = p / np.sum(p)

# 输出结果
teams = ['A', 'B', 'C', 'D']
print("团队强度参数 p_i：")
# 遍历对象下标i；后面j从i+1起避免重复一对。
for i in range(len(teams)):
    print(f"团队 {teams[i]} 的强度参数 p_{teams[i]} = {p[i]:.4f}")

print("\n归一化的团队强度参数 p_i (总和为1)：")
# 遍历对象下标i；后面j从i+1起避免重复一对。
for i in range(len(teams)):
    print(f"团队 {teams[i]} 的归一化强度参数 p_{teams[i]} = {p_normalized[i]:.4f}")


# %% 原单元 3 · markdown
# 五品牌比较表的行列记录胜出次数；每对共100次，10个无序对合计1000次。估计后用 pi_i/(pi_i+pi_j) 计算指定两品牌的偏好概率。
# - Paired Comparison Analysis
#
#     Suppose 100 consumers compare five tyre brands as follows:
#
#     | Brand   | Dunlop | Modi | Ceat | GoodYear | MRF |
#     |---------|--------|------|------|----------|-----|
#     | Dunlop  | -      | 80   | 59   | 52       | 77  |
#     | Modi    | 20     | -    | 60   | 46       | 56  |
#     | Ceat    | 41     | 40   | -    | 61       | 60  |
#     | G.Year  | 48     | 54   | 39   | -        | 67  |
#     | MRF     | 23     | 44   | 40   | 33       | -   |

# %% 原单元 4 · markdown
# 正强度pi_i>0，Pij=pi_i/(pi_i+pi_j)；令theta_i=log pi_i便等于sigmoid(theta_i-theta_j)。似然按无序对求和，若对有序i,j全求和就需防止重复计数。
# - 我们引入一个正的能力参数 $\pi_i\gt 0$，表示消费者对品牌 $i$ 的偏好强度。
# - 对于每一对不同的品牌 $(i,j)$，我们有以下的观测数据
#     - $n_{ij}$, 在比较中消费者选择品牌 $i$ 而不是品牌 $j$ 的次数；
#     - $n_{ji}$, 在比较中消费者选择品牌 $j$ 而不是品牌 $i$ 的次数；
#     - $n_{ij}+n_{ji}=n_{ij}^{total}$，两品牌被比较的总次数；
# - Bradley-Terry 模型的概率表示
#     - 假设品牌 $i$ 被偏好于品牌 $j$ 的概率 $P{ij}$
#         - $P_{ij}=\frac{\pi_i}{\pi_{i}+\pi_j}$
#     - 相应地
#         - $P_{ji}=\frac{\pi_j}{\pi_i+\pi_j}$
# - 基于成对比较数据和 Bradley-Terry 模型，我们可以构建对数似然函数
#     - $\ell(\pi)=\sum n_{ij}\log\frac{\pi_i}{\pi_i+\pi_j}+n_{ji}\log\frac{\pi_j}{\pi_i+\pi_j}$

# %% 原单元 5 · code
# data 的四元组依次为(i,j,n_ij,n_ji)。固定最后品牌的 log_pi4=0，优化前4个分数。函数逐对累加负对数似然，用 max-shift log-sum-exp 稳定计算指数和的对数，再把结果交给优化器。

# Brands and indices
brands = ['Dunlop', 'Modi', 'Ceat', 'GoodYear', 'MRF']
# enumerate同时给位置和品牌名。
brand_indices = {brand: idx for idx, brand in enumerate(brands)}

# Paired comparison data: (i, j, n_ij, n_ji)
data = [
    (0, 1, 80, 20),  # Dunlop vs Modi
    (0, 2, 59, 41),  # Dunlop vs Ceat
    (0, 3, 52, 48),  # Dunlop vs GoodYear
    (0, 4, 77, 23),  # Dunlop vs MRF
    (1, 2, 60, 40),  # Modi vs Ceat
    (1, 3, 46, 54),  # Modi vs GoodYear
    (1, 4, 56, 44),  # Modi vs MRF
    (2, 3, 61, 39),  # Ceat vs GoodYear
    (2, 4, 60, 40),  # Ceat vs MRF
    (3, 4, 67, 33)   # GoodYear vs MRF
]

# 定义负对数似然，供 SciPy 数值优化。
def neg_log_likelihood(log_pis):
    # log_pis is an array of length 4 (log_p0 to log_p3)
    # log_p4 is fixed at 0
    # 固定最后品牌log强度0，拼成长度5的完整数组。
    log_pis_full = np.append(log_pis, 0)  # Append log_p4 = 0
    LL = 0
    # 四元组解包：两个品牌下标与两方向观测次数。
    for i, j, n_ij, n_ji in data:
        log_pi = log_pis_full[i]
        log_pj = log_pis_full[j]
        # 两数取max供稳定log-sum-exp移位。
        max_log_pi_pj = max(log_pi, log_pj)
        # Compute log_sum_exp(log_pi, log_pj) in a numerically stable way
        # 先减max再exp，最后加回max，避免正指数overflow。
        log_sum_ij = max_log_pi_pj + np.log( np.exp(log_pi - max_log_pi_pj) + np.exp(log_pj - max_log_pi_pj) )
        # 累加当前比较对的binomial log likelihood（省略与参数无关常数）。
        LL += n_ij * (log_pi - log_sum_ij) + n_ji * (log_pj - log_sum_ij)
    # 取负号供最小化器使用；最大似然变为最小负似然。
    return -LL  # Negative log-likelihood


# %% 原单元 6 · code
# 从四维零向量 x0 开始，minimize(method=BFGS) 用数值差分估计梯度。result.success 表示优化器的退出状态；对估计分数指数化，再除以总和，得到统一尺度的强度。此处的 neg_log_likelihood 是五品牌任务重新定义的函数。
# Initial guesses for log_p0 to log_p3
# 四个自由log强度的起点；不是五个品牌强度本身。
x0 = np.zeros(4)

# Perform optimization
# 真实BFGS优化，未传jac时使用数值差分。
result = minimize(neg_log_likelihood, x0, method='BFGS')

# 只有成功时才报告估计参数。
if result.success:
    # 把固定为0的最后一维补回。
    log_pis_est = np.append(result.x, 0)  # Append log_p4 = 0
    # 把log强度还原为正强度。
    pis_est = np.exp(log_pis_est)
    # 把正强度归一化；成对概率仍需pi_i/(pi_i+pi_j)。
    pis_est_normalized = pis_est / np.sum(pis_est)
    print("Estimated Ability Scores:")
    # enumerate同时给位置和品牌名。
    for idx, brand in enumerate(brands):
        print(f"{brand}: {pis_est_normalized[idx]:.4f}")
else:
    print("Optimization failed:", result.message)


# %% 原单元 7 · code
# 这是一个空白代码单元，可留作补充计算。
