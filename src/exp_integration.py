import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ==================== 参数设置 ====================
# 字体设置
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = True

# 函数定义
def f(x):
    return np.exp(x)  # 测试函数：f(x) = e^x

# 积分区间
a, b = 0, 1

# 精确积分值
exact_value = np.exp(1) - 1  # ∫₀¹ e^x dx = e - 1

# 分割数设置
n = 32  # 基本分割数（前三个图的n值）
max_n_for_plot = 100  # 第四个图的最大n值

# 图像保存路径
save_path_1 = 'numerical_integration_results.png'

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


# ==================== 计算基本结果 ====================
print("=" * 60)
print(f"测试函数: f(x) = e^x, 区间: [{a}, {b}]")
print(f"精确积分值: {exact_value:.6f}")
print(f"分割数 n = {n}")
print("-" * 60)

# 计算三种方法的结果
left_val = left_rectangle(f, a, b, n)
mid_val = midpoint_rectangle(f, a, b, n)
trap_val = trapezoidal(f, a, b, n)

# 打印结果（保留三位小数）
print(f"左矩形法结果: {left_val:.3f}")
print(f"中矩形法结果: {mid_val:.3f}")
print(f"梯形法结果:  {trap_val:.3f}")
print("-" * 60)

# 计算误差
print(f"左矩形法误差: {abs(left_val - exact_value):.6f}")
print(f"中矩形法误差: {abs(mid_val - exact_value):.6f}")
print(f"梯形法误差:  {abs(trap_val - exact_value):.6f}")
print("=" * 60)

# ==================== 可视化部分 ====================
plt.figure(figsize=(14, 10))
plt.suptitle(f'图3：数值积分方法比较（以$f(x)=e^x$为例，n={n}, 精确值={exact_value:.4f}）', fontsize=16)

# ---------- 子图1：左矩形法 (n=32) ----------
plt.subplot(2, 2, 1)
x_fine = np.linspace(a, b, 400)
y_fine = f(x_fine)
x_coarse = np.linspace(a, b, n + 1)
dx = (b - a) / n

# 绘制函数曲线
plt.plot(x_fine, y_fine, 'b-', linewidth=2)

# 绘制矩形（左矩形）
for i in range(n):
    xi = a + i * dx
    rect = plt.Rectangle((xi, 0), dx, f(xi),
                         alpha=0.3, facecolor='red', edgecolor='black', linewidth=0.5)
    plt.gca().add_patch(rect)

# 标记分割点
plt.scatter(x_coarse[:-1], f(x_coarse[:-1]), color='red', s=20, zorder=5)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'左矩形法 (n={n})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0.02, 0.98, f'结果: {left_val:.4f}\n误差: {abs(left_val - exact_value):.6f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ---------- 子图2：中矩形法 (n=32) ----------
plt.subplot(2, 2, 2)
plt.plot(x_fine, y_fine, 'b-', linewidth=2)

# 绘制矩形（中矩形）
for i in range(n):
    xi_left = a + i * dx
    xi_right = a + (i + 1) * dx
    xi_mid = (xi_left + xi_right) / 2
    y_mid = f(xi_mid)

    rect = plt.Rectangle((xi_left, 0), dx, y_mid,
                         alpha=0.3, facecolor='orange', edgecolor='black', linewidth=0.5)
    plt.gca().add_patch(rect)
    plt.scatter(xi_mid, y_mid, color='red', s=20, zorder=5)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'中矩形法 (n={n})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0.02, 0.98, f'结果: {mid_val:.4f}\n误差: {abs(mid_val - exact_value):.6f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ---------- 子图3：梯形法 (n=32) ----------
plt.subplot(2, 2, 3)
plt.plot(x_fine, y_fine, 'b-', linewidth=2)

# 绘制梯形
for i in range(n):
    xi_left = a + i * dx
    xi_right = a + (i + 1) * dx
    y_left = f(xi_left)
    y_right = f(xi_right)

    # 绘制梯形填充
    vertices = [(xi_left, 0), (xi_right, 0),
                (xi_right, y_right), (xi_left, y_left)]
    polygon = plt.Polygon(vertices, alpha=0.3, facecolor='green',
                          edgecolor='darkgreen', linewidth=0.5)
    plt.gca().add_patch(polygon)

    # 标记端点
    plt.scatter([xi_left, xi_right], [y_left, y_right],
                color='blue', s=20, zorder=5)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'梯形法 (n={n})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.text(0.02, 0.98, f'结果: {trap_val:.4f}\n误差: {abs(trap_val - exact_value):.6f}',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ---------- 子图4：误差收敛曲线 ----------
plt.subplot(2, 2, 4)

# 生成不同n值下的误差数据
n_values = np.arange(2, max_n_for_plot + 1, 2)  # 偶数n值
errors_left = []
errors_mid = []
errors_trap = []

for n_i in n_values:
    errors_left.append(abs(left_rectangle(f, a, b, n_i) - exact_value))
    errors_mid.append(abs(midpoint_rectangle(f, a, b, n_i) - exact_value))
    errors_trap.append(abs(trapezoidal(f, a, b, n_i) - exact_value))

# 绘制误差收敛曲线
plt.loglog(n_values, errors_left, 'ro-', linewidth=1.5, markersize=4, label='左矩形法')
plt.loglog(n_values, errors_mid, 'bs-', linewidth=1.5, markersize=4, label='中矩形法')
plt.loglog(n_values, errors_trap, 'g^-', linewidth=1.5, markersize=4, label='梯形法')

plt.xlabel('分割数 n', fontsize=11)
plt.ylabel('绝对误差', fontsize=11)
plt.title('误差收敛特性 (对数坐标)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

# 添加参考线：1/n 和 1/n² 的收敛速率参考
ref_x = np.array([4, max_n_for_plot])
ref_y1 = 0.5 / ref_x  # O(1/n) 参考线
ref_y2 = 0.05 / (ref_x ** 2)  # O(1/n²) 参考线
plt.loglog(ref_x, ref_y1, 'k:', linewidth=1, alpha=0.7, label=r'$O(1/n)$')
plt.loglog(ref_x, ref_y2, 'k--', linewidth=1, alpha=0.7, label=r'$O(1/n^2)$')

plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
plt.show()

# ==================== 额外数据输出 ====================
print("\n不同分割数下的积分结果（保留三位小数）:")
print("-" * 50)
print(f"{'n':>4} | {'左矩形':>5} | {'中矩形':>5} | {'梯形法':>5} | {'精确值':>5}")
print("-" * 50)

for n_test in [4, 8, 16, 32, 64]:
    l_val = left_rectangle(f, a, b, n_test)
    m_val = midpoint_rectangle(f, a, b, n_test)
    t_val = trapezoidal(f, a, b, n_test)
    print(f"{n_test:4d} | {l_val:8.3f} | {m_val:8.3f} | {t_val:8.3f} | {exact_value:8.3f}")

print("=" * 60)
print(f"图像已保存为: {save_path_1}")
print("=" * 60)