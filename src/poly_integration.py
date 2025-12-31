import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ==================== 参数设置 ====================
# 字体设置
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = True


# 函数定义
def f(x):
    return x ** 2 + 0.5  # 定义示例函数：f(x) = x² + 0.5


# 积分区间
a, b = 0, 2

# 精确积分值
exact_integral = (b ** 3 / 3 + 0.5 * b) - (a ** 3 / 3 + 0.5 * a)

# 分割数设置
n_values_plot1 = [4, 8, 16]  # 图1的分割数列表
n_plot2 = 4  # 图2的分割数
n_range_plot1 = np.arange(2, 51)  # 图1误差曲线的n范围

# 图像保存路径
save_path_1 = 'riemann_sum_convergence.png'
save_path_2 = 'rect_vs_trap.png'


# ==================== 数值积分方法实现 ====================
def left_rectangle(f, a, b, n):
    """左矩形法"""
    dx = (b - a) / n
    total = sum(f(a + i * dx) for i in range(n))
    return total * dx


def midpoint_rectangle(f, a, b, n):
    """中矩形法"""
    dx = (b - a) / n
    total = sum(f(a + (i + 0.5) * dx) for i in range(n))
    return total * dx


def trapezoidal(f, a, b, n):
    """梯形法"""
    dx = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += 2 * f(a + i * dx)
    return total * dx / 2


# ==================== 图1：黎曼和思想 ====================
plt.figure(figsize=(14, 10))
plt.suptitle('图1：黎曼和思想与数值积分逼近（以$f(x)=x^2+0.5$为例）', fontsize=16)

# 三个子图展示不同分割数
for idx, n in enumerate(n_values_plot1, 1):
    plt.subplot(2, 2, idx)

    # 生成数据
    x = np.linspace(a, b, 400)
    y = f(x)
    x_approx = np.linspace(a, b, n + 1)
    y_approx = f(x_approx)
    dx = (b - a) / n

    # 绘制函数曲线
    plt.plot(x, y, 'b-', linewidth=2)

    # 绘制矩形（左矩形）
    for i in range(n):
        xi = a + i * dx
        plt.bar(xi, f(xi), width=dx, alpha=0.3,
                color='r', edgecolor='black', align='edge')

    # 标记分割点
    plt.scatter(x_approx, y_approx, color='red', s=30, zorder=5)

    # 计算近似积分值（左矩形）
    approx_integral = left_rectangle(f, a, b, n)

    plt.title(f'黎曼和逼近 (n={n})', fontsize=12)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    # 显示积分值对比
    plt.text(0.05, 0.95, f'精确值: {exact_integral:.3f}\n近似值: {approx_integral:.3f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 第四个图：误差收敛趋势
plt.subplot(2, 2, 4)
errors = []

for n in n_range_plot1:
    approx = left_rectangle(f, a, b, n)
    errors.append(abs(approx - exact_integral))

plt.loglog(n_range_plot1, errors, 'bo-', linewidth=2, markersize=4)
plt.xlabel('分割数 n', fontsize=11)
plt.ylabel('绝对误差', fontsize=11)
plt.title('误差随分割数的变化')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
plt.show()

# ==================== 图2：矩形法与梯形法比较 ====================
plt.figure(figsize=(12, 5))

# 子图1：矩形法（中矩形）
plt.subplot(1, 2, 1)
n = n_plot2
x = np.linspace(a, b, 400)
y = f(x)
x_approx = np.linspace(a, b, n + 1)
dx = (b - a) / n

# 绘制函数曲线
plt.plot(x, y, 'b-', linewidth=2)

# 绘制矩形（中矩形法）
for i in range(n):
    xi_left = a + i * dx
    xi_right = a + (i + 1) * dx
    xi_mid = (xi_left + xi_right) / 2
    y_mid = f(xi_mid)

    # 绘制矩形
    rect = plt.Rectangle((xi_left, 0), dx, y_mid,
                         alpha=0.3, color='orange', edgecolor='red', linewidth=1.5)
    plt.gca().add_patch(rect)

    # 标记中点
    plt.scatter(xi_mid, y_mid, color='red', s=50, zorder=5)

    # 连接线
    plt.plot([xi_left, xi_right], [y_mid, y_mid], 'r-', linewidth=1.5)

# 标记分割点
plt.scatter(x_approx, f(x_approx), color='blue', s=40, zorder=5)

# 计算中矩形法积分
midpoint_integral = midpoint_rectangle(f, a, b, n)

plt.title('中矩形法 (Midpoint Rule)', fontsize=13)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.text(0.05, 0.95, f'积分值: {midpoint_integral:.4f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 子图2：梯形法
plt.subplot(1, 2, 2)
# 绘制函数曲线
plt.plot(x, y, 'b-', linewidth=2)

# 绘制梯形
for i in range(n):
    xi_left = a + i * dx
    xi_right = a + (i + 1) * dx
    y_left = f(xi_left)
    y_right = f(xi_right)

    # 填充梯形区域
    vertices = [(xi_left, 0), (xi_right, 0),
                (xi_right, y_right), (xi_left, y_left)]
    polygon = plt.Polygon(vertices, alpha=0.3, color='green', edgecolor='darkgreen', linewidth=1.5)
    plt.gca().add_patch(polygon)

    # 绘制梯形上边界
    plt.plot([xi_left, xi_right], [y_left, y_right], 'darkgreen', linewidth=1.5)

# 标记端点
plt.scatter(x_approx, f(x_approx), color='blue', s=40, zorder=5)

# 计算梯形法积分
trapezoidal_integral = trapezoidal(f, a, b, n)

plt.title('梯形法 (Trapezoidal Rule)', fontsize=13)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.text(0.05, 0.95, f'积分值: {trapezoidal_integral:.4f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle(f'图2：矩形法与梯形法比较（以$f(x)=x^2+0.5$为例，n={n}, 精确值={exact_integral:.4f}）', fontsize=14)
plt.tight_layout()
plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
plt.show()

# ==================== 额外数据输出 ====================
print("=" * 60)
print("图像已保存为：")
print(f"1. {save_path_1}")
print(f"2. {save_path_2}")
print("=" * 60)
print(f"精确积分值: {exact_integral:.6f}")
print(f"中矩形法近似值 (n={n}): {midpoint_integral:.6f}")
print(f"梯形法近似值 (n={n}): {trapezoidal_integral:.6f}")
print(f"中矩形法误差: {abs(midpoint_integral - exact_integral):.6f}")
print(f"梯形法误差: {abs(trapezoidal_integral - exact_integral):.6f}")
print("=" * 60)
