import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os  # <--- 新增：用于路径处理

def plot_history_metrics(history_metrics):
    """
    绘制历史指标，自动处理因延迟导致的长度不一致问题。
    """
    # 1. 获取基准长度（通常以 Jitter 列表为准，它是全长的）
    n_rounds = len(history_metrics['J_t_x'])
    rounds = np.arange(n_rounds)

    # 2. 定义辅助函数：对齐数据长度
    def align_data(data_list, target_length):
        current_len = len(data_list)
        if current_len == target_length:
            return data_list
        elif current_len < target_length:
            # 假设缺失的是开头的数据（因为延迟）
            missing_count = target_length - current_len
            # 在列表前面填充 NaN
            return [np.nan] * missing_count + data_list
        else:
            # 如果数据比时间轴还长（不太可能，但为了健壮性），截断
            return data_list[:target_length]

    # 3. 对所有 C_t 数据进行对齐处理
    ct_x_aligned = align_data(history_metrics['C_t_x'], n_rounds)
    ct_z_aligned = align_data(history_metrics['C_t_z'], n_rounds)
    ct_ens_aligned = align_data(history_metrics['C_t_ens'], n_rounds)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: Jitter (J_t) ---
    ax1 = axes[0]
    ax1.plot(rounds, history_metrics['J_t_x'], label='J_t (X)', color='#1f77b4', alpha=0.8)
    ax1.plot(rounds, history_metrics['J_t_z'], label='J_t (Z)', color='#2ca02c', alpha=0.8)
    ax1.plot(rounds, history_metrics['J_t_ens'], label='J_t (Ensemble)', color='#ff7f0e', alpha=0.8)

    ax1.set_ylabel('Jitter ($J_t$)', fontsize=12)
    ax1.set_title('Jitter Metric Evolution', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 2: Momentum (C_t) ---
    ax2 = axes[1]
    # 检查列表是否非空，避免报错
    if len(ct_x_aligned) > 0:
        ax2.plot(rounds, ct_x_aligned, label='C_t (X)', color='#1f77b4', alpha=0.8)
    if len(ct_z_aligned) > 0:
        ax2.plot(rounds, ct_z_aligned, label='C_t (Z)', color='#2ca02c', alpha=0.8)
    if len(ct_ens_aligned) > 0:
        ax2.plot(rounds, ct_ens_aligned, label='C_t (Ensemble)', color='#ff7f0e', alpha=0.8)

    ax2.set_xlabel('Rounds (Time)', fontsize=12)
    ax2.set_ylabel('Momentum ($C_t$)', fontsize=12)
    ax2.set_title('Momentum Trajectory Evolution (Delayed)', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# 调用方法：
# plot_history_metrics(history_metrics)

# ==========================================
# 新增绘图函数 (添加到 OLSMF_Delay_Model 类外部或内部均可，这里建议放在辅助函数区域)
# ==========================================
def plot_momentum_trajectory(history_metrics, show_x=False, show_z=False, show_ens=True):
    """
    绘制动量/兼容性分数 C_t 的历史轨迹
    :param history_metrics: 包含 C_t_x, C_t_z, C_t_ens 的字典
    :param show_x: 是否展示 X 模态 (由参数控制)
    :param show_z: 是否展示 Z 模态 (由参数控制)
    :param show_ens: 是否展示 Ensemble 模态 (默认展示)
    """
    plt.figure(figsize=(12, 6))

    # 获取数据长度以生成 x 轴
    rounds = range(len(history_metrics['C_t_ens']))

    # 绘制 X 模态 (可选)
    if show_x:
        data = history_metrics.get('C_t_x', [])
        if data:
            plt.plot(rounds, data, label='Momentum X (C_t)', alpha=0.5, linestyle='--', color='blue')

    # 绘制 Z 模态 (可选)
    if show_z:
        data = history_metrics.get('C_t_z', [])
        if data:
            plt.plot(rounds, data, label='Momentum Z (C_t)', alpha=0.5, linestyle='--', color='green')

    # 绘制 Ensemble 模态 (默认展示)
    if show_ens:
        data = history_metrics.get('C_t_ens', [])
        if data:
            plt.plot(rounds, data, label='Momentum Ensemble (C_t)', linewidth=2.5, color='red')

    plt.title("Momentum Trajectory (Compatibility Score) over Time")
    plt.xlabel("Rounds")
    plt.ylabel("Momentum Score (C_t)")
    plt.ylim(-0.1, 1.1)  # 稍微留出边界
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_learning_dynamics(history_metrics, delay_len, threshold, window_len, show_x=False, show_z=False):
    """
    Plot learning dynamics with customized title showing parameters.
    """
    # 1. Get Data
    j_data_ens = history_metrics.get('J_t_ens', [])
    c_data_ens = history_metrics.get('C_t_ens', [])

    # 2. Generate X axis
    # Jitter is recorded from start
    rounds_jitter = range(len(j_data_ens))

    # Momentum is delayed
    # Delay = Total rounds - Momentum rounds
    delay = len(j_data_ens) - len(c_data_ens)
    rounds_momentum = range(delay, len(j_data_ens))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Add Super Title with Parameters
    fig.suptitle(f'Learning Dynamics\nDelay: {delay_len} | Threshold: {threshold} | Window: {window_len}',
                 fontsize=14, fontweight='bold')

    # -----------------------------
    # Subplot 1: Jitter (Full duration)
    # -----------------------------
    if show_x:
        ax1.plot(rounds_jitter, history_metrics.get('J_t_x', []), label='Jitter X',
                 alpha=0.4, linestyle='--', color='blue')
    if show_z:
        ax1.plot(rounds_jitter, history_metrics.get('J_t_z', []), label='Jitter Z',
                 alpha=0.4, linestyle='--', color='green')

    ax1.plot(rounds_jitter, j_data_ens, label='Jitter Ensemble',
             linewidth=2, color='red')

    ax1.set_ylabel('Jitter ($J_t$)')
    ax1.set_title('Jitter Dynamics (Stability)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # -----------------------------
    # Subplot 2: Momentum (Delayed)
    # -----------------------------
    if show_x:
        c_data_x = history_metrics.get('C_t_x', [])
        if len(c_data_x) == len(rounds_momentum):
            ax2.plot(rounds_momentum, c_data_x, label='Momentum X',
                     alpha=0.4, linestyle='--', color='blue')

    if show_z:
        c_data_z = history_metrics.get('C_t_z', [])
        if len(c_data_z) == len(rounds_momentum):
            ax2.plot(rounds_momentum, c_data_z, label='Momentum Z',
                     alpha=0.4, linestyle='--', color='green')

    # Ensemble
    if len(c_data_ens) > 0:
        ax2.plot(rounds_momentum, c_data_ens, label='Momentum Ensemble',
                 linewidth=2, color='red')

    ax2.set_ylabel('Momentum ($C_t$)')
    ax2.set_xlabel('Rounds')
    ax2.set_title('Momentum/Compatibility Trajectory')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()


def plot_gradient_curves(delayed_grads, hint_grads, save_dir, filename="gradient_curves.png", window_size=50):
    """
    绘制并保存梯度范数曲线

    Args:
        delayed_grads: 延迟更新的梯度列表
        hint_grads: Hint更新的梯度列表
        save_dir: 保存目录
        filename: 文件名
        window_size: 滑动平均窗口大小，用于平滑曲线
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 6))

    # 辅助函数：计算滑动平均
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window), 'valid') / window

    # 绘制原始数据（透明度低）和滑动平均（实线）
    steps = range(len(delayed_grads))

    # Delayed Gradients
    plt.plot(steps, delayed_grads, alpha=0.2, color='blue', label='Delayed Grad (Raw)')
    if len(delayed_grads) > window_size:
        smooth_delayed = moving_average(delayed_grads, window_size)
        plt.plot(steps[window_size - 1:], smooth_delayed, color='blue', linewidth=2,
                 label=f'Delayed Grad (MA-{window_size})')

    # Hint Gradients
    plt.plot(steps, hint_grads, alpha=0.2, color='orange', label='Hint Grad (Raw)')
    if len(hint_grads) > window_size:
        smooth_hint = moving_average(hint_grads, window_size)
        plt.plot(steps[window_size - 1:], smooth_hint, color='orange', linewidth=2,
                 label=f'Hint Grad (MA-{window_size})')

    plt.title('Gradient Norms over Time (Online Learning)')
    plt.xlabel('Time Step')
    plt.ylabel('Gradient L2 Norm')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Gradient curves saved to {save_path}")


# ==========================================
# 1. 新增 Loss 绘图函数 (放在 plot_gradient_curves 旁边)
# ==========================================
def plot_loss_curves(delayed_losses, hint_losses, save_dir, filename="loss_curves.png", window_size=50):
    """
    绘制并保存 Loss 变化曲线

    Args:
        delayed_losses: 延迟更新的 Loss 列表
        hint_losses: Hint 更新的 Loss 列表
        save_dir: 保存目录
        filename: 文件名
        window_size: 滑动平均窗口大小
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 6))

    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window), 'valid') / window

    steps = range(len(delayed_losses))

    # Delayed Losses
    # 只绘制大于0的部分（即发生了更新的时刻），避免0值拉低平均线（可选，这里保留原始数据但做平滑）
    plt.plot(steps, delayed_losses, alpha=0.15, color='green', label='Delayed Loss (Raw)')
    if len(delayed_losses) > window_size:
        smooth_delayed = moving_average(delayed_losses, window_size)
        plt.plot(steps[window_size - 1:], smooth_delayed, color='green', linewidth=2,
                 label=f'Delayed Loss (MA-{window_size})')

    # Hint Losses
    plt.plot(steps, hint_losses, alpha=0.15, color='red', label='Hint Loss (Raw)')
    if len(hint_losses) > window_size:
        smooth_hint = moving_average(hint_losses, window_size)
        plt.plot(steps[window_size - 1:], smooth_hint, color='red', linewidth=2, label=f'Hint Loss (MA-{window_size})')

    plt.title('Training Loss over Time (Online Learning)')
    plt.xlabel('Time Step')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Loss curves saved to {save_path}")

