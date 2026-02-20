import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os  # <--- 新增这行导入

class DriftEvaluator:
    def __init__(self, y_true, results_dict, drift_start, delay_rounds, dataset,window_size=200):
        """
        初始化评估器
        :param y_true: 真实标签数组 (numpy array)
        :param results_dict: 结果字典 {'MethodName': {'errors': [], 'preds': []}, ...}
        :param drift_start: 漂移开始的索引 (int)
        :param delay_rounds: 延迟步数 D (int)
        :param window_size: 滑动窗口大小 w (int)
        """
        self.y_true = np.array(y_true)
        self.results = results_dict
        self.drift_start = drift_start
        self.delay_rounds = delay_rounds
        self.window_size = window_size
        self.total_steps = len(y_true)
        self.dataset=dataset

        # 延迟结束点
        self.delay_end = self.drift_start + self.delay_rounds

    def calculate_sliding_accuracy(self, error_list):
        """
        计算滑动窗口准确率
        Acc = 1 - MovingAverage(Error)
        """
        # 将 error (0/1) 转换为 accuracy (1/0)
        correctness = 1 - np.array(error_list)
        # 使用 Pandas Rolling 计算滑动平均
        series = pd.Series(correctness)
        # min_periods=1 保证初期也有数据
        rolling_acc = series.rolling(window=self.window_size, min_periods=1).mean()#对于时刻t（drift_start），滑动窗口计算的是区间[t-W+1, t] 的平均值
        return rolling_acc.values

    def get_metrics_table(self):
        """
        计算核心量化指标：CER, DPAA, Recovery Steps
        """
        metrics_summary = {}

        # 1. 计算漂移前的稳定准确率 (作为恢复的目标基准)
        # 取漂移前 window_size 长度的平均值
        pre_drift_start = max(0, self.drift_start - self.window_size)

        for method_name, data in self.results.items():
            errors = np.array(data['errors'])
            correctness = 1 - errors
            sliding_acc = self.calculate_sliding_accuracy(errors)

            # --- 动态计算该方法的 Baseline ---
            # 取漂移前一段窗口的平均值作为该方法的"正常水平"
            if self.drift_start > 0:
                baseline_slice = correctness[pre_drift_start: self.drift_start]
                baseline_acc = np.mean(baseline_slice) if len(baseline_slice) > 0 else 0.5
            else:
                baseline_acc = 0.5  # 默认值

            # --- 指标 1: 全局 CER ---
            # OSLMF 原文指标
            cer = np.cumsum(errors) / (np.arange(len(errors)) + 1)
            final_cer = cer[-1]

            # --- 指标 2: 延迟期平均准确率 (DPAA) ---
            # 统计区间: [drift_start, drift_start + delay]
            # 这里的切片要小心索引越界
            safe_end = min(self.delay_end, len(correctness))
            dpaa_segment = correctness[self.drift_start: safe_end]
            dpaa = np.mean(dpaa_segment) if len(dpaa_segment) > 0 else 0.0

            # --- 指标 3: 鲁棒性指标 - 最大跌幅 (Max Drop) ---
            # 在漂移后到延迟结束这段最困难的时间里，准确率最低跌到了多少
            # Drop = Baseline - Min_Accuracy_in_Delay_Period
            drift_period_acc = sliding_acc[self.drift_start: min(self.delay_end + self.window_size, self.total_steps)]
            min_acc_during_drift = np.min(drift_period_acc) if len(drift_period_acc) > 0 else 0.0
            max_drop = max(0, baseline_acc - min_acc_during_drift)

            # --- 指标 3: 恢复步数 (Recovery Steps) ---
            # 目标: 恢复到漂移前准确率的 95%
            recovery_target = baseline_acc * 0.95

            recovery_steps = float('inf')  # 默认未恢复

            # 标记是否已经发生了性能下降
            has_dropped = False

            # 从漂移点开始向后搜索
            # 为了避免噪声导致的瞬间反弹，我们要求连续 10 步都高于阈值
            stability_window = 10

            # 只在漂移发生后搜索
            search_start_idx = self.drift_start

            # 优化：直接遍历
            for i in range(search_start_idx, self.total_steps - stability_window):
                current_acc = sliding_acc[i]

                # 阶段 1: 确认性能下降
                if not has_dropped:
                    # 如果准确率低于目标值，标记为已下降，开始寻找恢复
                    if current_acc < recovery_target:
                        has_dropped = True
                    else:
                        # 如果一直没掉下来，且已经过了很远（比如过了2倍窗口大小），说明该模型抵抗力极强
                        # 这种情况下，恢复步数可以说是 0
                        if i > self.drift_start + self.window_size:
                            recovery_steps = 0
                            break
                    continue  # 继续下一轮循环

                # 阶段 2: 寻找恢复 (只有在 has_dropped = True 后才执行)
                if has_dropped:
                    # 检查当前点是否回升达标
                    if current_acc >= recovery_target:
                        # 检查后续 stability_window 步的均值/最小值是否也达标 (防止假反弹)
                        future_window = sliding_acc[i: i + stability_window]
                        if np.min(future_window) >= recovery_target:
                            recovery_steps = i - self.drift_start
                            break

            metrics_summary[method_name] = {
                "CER": f"{final_cer:.4f}",
                "DPAA": f"{dpaa:.4f}",
                "Max Drop": f"{max_drop:.4f}",  # 新增：跌幅越小越好
                "Recov. Steps": recovery_steps if recovery_steps != float('inf') else ">Max",
                "Baseline": f"{baseline_acc:.4f}"
            }

        return pd.DataFrame(metrics_summary).T

    def plot_curves(self, title="Prequential Accuracy Analysis",BATCH_SIZE=0,WINDOW_SIZE=0,save_dir=None,metrics_table=None):
        """
        绘制论文级别的对比图
        """

        if metrics_table==None:
        # 1. 预先计算指标表
            metrics_df = self.get_metrics_table()
        else:
            metrics_df=metrics_table

        # 创建双子图共享 x 轴
        fig, (ax_acc, ax_cer) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                             gridspec_kw={'height_ratios': [2, 1]})

        # 定义通用绘图元素
        # 阴影区 (Label Delay Period) - 在两个图上都画
        for ax in [ax_acc, ax_cer]:
            ax.axvspan(self.drift_start, self.delay_end, color='#d9d9d9', alpha=0.5, zorder=0)
            ax.axvline(x=self.drift_start, color='black', linestyle='--', linewidth=1.5, zorder=1)

        # 4. 绘制各方法的曲线
        # 保留你原有的颜色定义
        colors = {'X': 'red', 'Z': 'blue', 'Ens': 'green','soft_Ens':'#d62728','hard_Ens':'#800020'}
        styles = {'X': '--', 'Z': '-.', 'Ens': '-','soft_Ens': ':','hard_Ens': '-.'}

        new_colors = {
            'origin_X': '#9467bd',  # Muted Purple
            'origin_Z': '#ff7f0e',  # Safety Orange
            'origin_Ens': '#17becf'  # Cyan-Teal
        }

        # 2. 新增线型 (使用 Tuple 自定义虚线模式)
        # 格式: (offset, (on_len, off_len, on_len, off_len, ...))
        new_styles = {
            'origin_X': (0, (3, 1, 1, 1, 1, 1)),  # Dash-dot-dot (划-点-点)
            'origin_Z': (0, (5, 5)),  # Loosely dashed (长虚线，间隔大)
            'origin_Ens': (0, (1, 1))  # Densely dotted (密集点线)
        }
        # 合并字典
        colors.update(new_colors)
        styles.update(new_styles)


        # 3. 绘制各方法的曲线
        # colors = {'X': 'red', 'Z': 'blue', 'Ens': 'green'}
        # styles = {'X': '--', 'Z': '-.', 'Ens': '-'}
        # 获取默认颜色循环，防止方法名不在colors字典中时颜色重复
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']

        for i, (name, data) in enumerate(self.results.items()):
            errors = data['errors']
            #滑动准确率
            acc_curve = self.calculate_sliding_accuracy(errors)
            #累计错误率 (CER)
            cer_curve = np.cumsum(errors) / (np.arange(len(errors)) + 1)

            # --- 核心修改：从指标表中获取数据并添加到图例 ---
            if name in metrics_df.index:
                rec_steps = metrics_df.loc[name, "Recov. Steps"]
                dpaa = metrics_df.loc[name, "DPAA"]
                # 格式化图例：Method (DPAA: 0.xx, Rec: xxx)
                label_str = f"{name} (DPAA:{dpaa}, Rec:{rec_steps})"
            else:
                label_str = name

            # --- 样式处理 ---
            c = colors.get(name, default_colors[i % len(default_colors)])
            s = styles.get(name, '-')
            lw = 2.5 if "Ours" in name else 1.5
            alpha = 1.0 if "Ours" in name else 0.8

            # --- 绘图 1: 滑动准确率 (Top) ---
            ax_acc.plot(acc_curve, label=label_str, color=c, linestyle=s, linewidth=lw, alpha=alpha)

            # --- 绘图 2: CER (Bottom) ---
            # CER 只需要简单的标签，因为详细指标在上图展示了
            ax_cer.plot(cer_curve, label=name, color=c, linestyle=s, linewidth=lw, alpha=alpha)

        # ====================
        # 上图装饰 (Accuracy)
        # ====================
        ax_acc.set_title(title, fontsize=14, fontweight='bold')
        ax_acc.set_ylabel(f"Prequential Accuracy (w={self.window_size})", fontsize=12)
        ax_acc.set_ylim(0.0, 1.05)
        ax_acc.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
        ax_acc.grid(True, linestyle=':', alpha=0.6)

        # 添加文本框信息
        info_text = (f"Dataset: {self.dataset}\n"
                     f"Drift Start: {self.drift_start}\n"
                     f"Delay Rounds: {self.delay_rounds}\n"
                     f"Window Size: {self.window_size}")
        ax_acc.text(0.02, 0.02, info_text, transform=ax_acc.transAxes,
                    fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # ====================
        # 下图装饰 (CER)
        # ====================
        ax_cer.set_ylabel("Cumulative Error Rate (CER)", fontsize=12)
        ax_cer.set_xlabel("Time Steps (Instances)", fontsize=12)
        ax_cer.grid(True, linestyle=':', alpha=0.6)
        # CER 的范围通常在 [0, 1] 之间，但为了看清细节，可以不强行固定 ylim，或者根据数据自动调整

        # 标注漂移点含义
        ax_acc.text(self.drift_start, 1.02, "Drift Start", ha='center', va='bottom',
                    fontsize=9, color='black', transform=ax_acc.get_xaxis_transform())
        ax_acc.text(self.delay_end, 1.02, "Labels Arrive", ha='center', va='bottom',
                    fontsize=9, color='gray', transform=ax_acc.get_xaxis_transform())


        # 聚焦到漂移前后展示 (Zoom-in)
        # 展示范围: 漂移前 1000 步 ~ 漂移后 2000 步
        if self.drift_start==0:
            view_start=0
            view_end=self.total_steps
        else:
            # view_start = max(0, self.drift_start - 1000)
            # view_end = min(self.total_steps, self.delay_end + 1500)
            view_start = 0
            view_end = self.total_steps
        plt.xlim(view_start, view_end)

        plt.tight_layout()

        # --- 核心修改：自动生成文件名并保存 ---
        if save_dir:

            if "LACH" not in save_dir:
                file_format = "png"
                # 1. 自动构建文件名: {Dataset}_Drift{Start}_Delay{Delay}_Win{Win}.png
                # 使用 safe_filename 处理可能的非法字符
                safe_dataset = str(self.dataset).replace('/', '_').replace('\\', '_')
                file_name = (f"{safe_dataset}_"
                             f"Drift{self.drift_start}_"
                             f"Delay{self.delay_rounds}_"
                             f"Win{self.window_size}_"
                             f"BATCH_SIZE{BATCH_SIZE}_"
                             f"WINDOWS_SIZE{WINDOW_SIZE}.{file_format}")
            else:
                file_format = "png"
                # 1. 自动构建文件名: {Dataset}_Drift{Start}_Delay{Delay}_Win{Win}.png
                # 使用 safe_filename 处理可能的非法字符
                file_name = ("LACH"+f".{file_format}")


            # 2. 组合完整路径
            full_path = os.path.join(save_dir, file_name)

            # 3. 检查并创建目录
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"[Info] Created directory: {save_dir}")
                except OSError as e:
                    print(f"[Error] Failed to create directory {save_dir}: {e}")

            # 4. 保存文件 (自动覆盖)
            try:
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"[Success] Auto-saved figure to: {full_path}")
            except Exception as e:
                print(f"[Error] Failed to save figure: {e}")
        plt.show()


if __name__ == "__main__":
    # 配置
    # N_SAMPLES = 3000
    DRIFT_START = 1000
    # DELAY = 500
    # WINDOW = 100  # 小窗口以突显变化
    #
    # # 模拟真实标签 (不重要，主要看 errors)
    # y_true = np.ones(N_SAMPLES)
    #
    # # --- 模拟 Wait-and-See (被动等待) ---
    # # 漂移前: 错误率低 (0.1)
    # # 延迟期: 错误率极高 (0.8) -> 模拟"瞎了"
    # # 延迟后: 缓慢恢复 (0.2)
    # errors_base = np.concatenate([
    #     np.random.choice([0, 1], DRIFT_START, p=[0.9, 0.1]),
    #     np.random.choice([0, 1], DELAY, p=[0.2, 0.8]),
    #     np.random.choice([0, 1], N_SAMPLES - DRIFT_START - DELAY, p=[0.8, 0.2])
    # ])
    #
    # # --- 模拟 Ours (主动适应) ---
    # # 漂移前: 错误率低 (0.1)
    # # 延迟期: 错误率略有升高但依然不错 (0.2) -> 模拟利用伪标签适应
    # # 延迟后: 恢复优秀 (0.1)
    # errors_ours = np.concatenate([
    #     np.random.choice([0, 1], DRIFT_START, p=[0.9, 0.1]),
    #     np.random.choice([0, 1], DELAY, p=[0.8, 0.2]),
    #     np.random.choice([0, 1], N_SAMPLES - DRIFT_START - DELAY, p=[0.9, 0.1])
    # ])
    #
    # results = {
    #     'Wait-and-See': {'errors': errors_base, 'preds': []},
    #     'Ours': {'errors': errors_ours, 'preds': []}
    # }
    #
    # # 初始化评估器
    # evaluator = DriftEvaluator(
    #     y_true=y_true,
    #     results_dict=results,
    #     drift_start=DRIFT_START,
    #     delay_rounds=DELAY,
    #     dataset="Test_Simulation",
    #     window_size=WINDOW
    # )
    #
    # # 1. 打印指标表
    # print("\n=== Performance Metrics ===")
    # print(evaluator.get_metrics_table())
    #
    # # 2. 绘制曲线 (不保存)
    # print("\n=== Plotting Curves ===")
    # evaluator.plot_curves(save_dir="./test_output")  # 尝试保存到当前目录下的 test_output