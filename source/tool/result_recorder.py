# 文件：result_recorder.py
"""
评测结果记录器
用于保存DriftEvaluator的评估结果到文件，便于后续分析
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from matplotlib import pyplot as plt


class ResultRecorder:
    def __init__(self, base_save_dir):
        """
        初始化结果记录器

        Args:
            base_save_dir: 基础保存目录，与plot_curves的save_dir一致
        """
        self.base_save_dir = base_save_dir

    def generate_filename_base(self, dataset, drift_start, delay_rounds, window_size,
                               batch_size=None, window_param=None):
        """
        生成与plot_curves一致的文件名基础部分

        Args:
            参数与DriftEvaluator中的参数一致
        """
        # 处理数据集名称中的特殊字符
        safe_dataset = str(dataset).replace('/', '_').replace('\\', '_')

        # 构建基础文件名
        filename_parts = [
            f"{safe_dataset}",
            f"Drift{drift_start}",
            f"Delay{delay_rounds}",
            f"Win{window_size}"
        ]

        # 可选参数
        if batch_size is not None:
            filename_parts.append(f"BATCH_SIZE{batch_size}")
        if window_param is not None:
            filename_parts.append(f"WINDOWS_SIZE{window_param}")

        return "_".join(filename_parts)

    def save_metrics_table(self, metrics_df, dataset, drift_start, delay_rounds,
                           window_size, batch_size=None, window_param=None,
                           file_format="csv"):
        """
        保存指标表格到文件

        Args:
            metrics_df: get_metrics_table()返回的DataFrame
            file_format: 保存格式，支持csv、json、excel
        """
        # 确保目录存在
        os.makedirs(self.base_save_dir, exist_ok=True)

        # 生成基础文件名
        filename_base = self.generate_filename_base(
            dataset, drift_start, delay_rounds, window_size, batch_size, window_param
        )

        # 添加时间戳以确保唯一性
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_metrics_{timestamp}"

        # 根据格式保存
        if file_format.lower() == "csv":
            filepath = os.path.join(self.base_save_dir, f"{filename}.csv")
            metrics_df.to_csv(filepath)
            print(f"[Success] Metrics saved to CSV: {filepath}")

        elif file_format.lower() == "json":
            filepath = os.path.join(self.base_save_dir, f"{filename}.json")

            # 将DataFrame转换为字典格式
            metrics_dict = {
                "metadata": {
                    "dataset": dataset,
                    "drift_start": drift_start,
                    "delay_rounds": delay_rounds,
                    "window_size": window_size,
                    "batch_size": batch_size,
                    "window_param": window_param,
                    "timestamp": timestamp
                },
                "metrics": metrics_df.to_dict(orient="index")
            }

            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            print(f"[Success] Metrics saved to JSON: {filepath}")

        elif file_format.lower() in ["xlsx", "excel"]:
            filepath = os.path.join(self.base_save_dir, f"{filename}.xlsx")
            metrics_df.to_excel(filepath)
            print(f"[Success] Metrics saved to Excel: {filepath}")

        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return filepath

    def save_detailed_results(self, results_dict, y_true, dataset, drift_start,
                              delay_rounds, window_size, batch_size=None,
                              window_param=None, save_probs=True):
        """
        保存详细的评估结果，包括每个时间步的预测和错误

        Args:
            results_dict: DriftEvaluator使用的results_dict格式
            y_true: 真实标签
            save_probs: 是否保存概率值（如果存在）
        """
        # 确保目录存在
        os.makedirs(self.base_save_dir, exist_ok=True)

        # 生成基础文件名
        filename_base = self.generate_filename_base(
            dataset, drift_start, delay_rounds, window_size, batch_size, window_param
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_detailed_{timestamp}"
        filepath = os.path.join(self.base_save_dir, f"{filename}.csv")

        # 创建DataFrame来存储详细结果
        detailed_data = []

        # 遍历所有时间步
        n_steps = len(y_true)

        # 获取第一个方法的结果长度作为参考
        first_method = list(results_dict.keys())[0]
        n_results = len(results_dict[first_method]['errors'])

        # 确保长度一致
        n_steps = min(n_steps, n_results)

        for t in range(n_steps):
            row = {
                "time_step": t,
                "y_true": y_true[t] if t < len(y_true) else np.nan,
                "drift_region": self._get_drift_region(t, drift_start, delay_rounds)
            }

            # 添加每个方法的预测结果
            for method_name, data in results_dict.items():
                # 清理列名中的特殊字符
                safe_method_name = method_name.replace(' ', '_').replace('-', '_')

                if t < len(data['errors']):
                    row[f"{safe_method_name}_error"] = data['errors'][t]
                    row[f"{safe_method_name}_pred"] = data['preds'][t] if 'preds' in data and t < len(
                        data['preds']) else np.nan

                    # 如果存在概率值且需要保存
                    if save_probs and 'probs' in data and t < len(data['probs']):
                        row[f"{safe_method_name}_prob"] = data['probs'][t]

            detailed_data.append(row)

        # 创建DataFrame并保存
        df_detailed = pd.DataFrame(detailed_data)
        df_detailed.to_csv(filepath, index=False)

        print(f"[Success] Detailed results saved to: {filepath}")
        return filepath

    def _get_drift_region(self, t, drift_start, delay_rounds):
        """确定当前时间步所属的漂移区域"""
        if t < drift_start:
            return "pre_drift"
        elif t < drift_start + delay_rounds:
            return "delay_period"
        else:
            return "post_drift"

    def save_summary_report(self, metrics_df, dataset, drift_start, delay_rounds,
                            window_size, batch_size=None, window_param=None):
        """
        保存汇总报告，包含关键统计信息
        """
        # 确保目录存在
        os.makedirs(self.base_save_dir, exist_ok=True)

        # 生成基础文件名
        filename_base = self.generate_filename_base(
            dataset, drift_start, delay_rounds, window_size, batch_size, window_param
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_summary_{timestamp}.txt"
        filepath = os.path.join(self.base_save_dir, filename)

        # 创建汇总报告
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DRIFT EVALUATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Drift Start: {drift_start}\n")
            f.write(f"Delay Rounds: {delay_rounds}\n")
            f.write(f"Window Size: {window_size}\n")
            if batch_size is not None:
                f.write(f"Batch Size: {batch_size}\n")
            if window_param is not None:
                f.write(f"Window Param: {window_param}\n")
            f.write(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")

            # 将DataFrame转换为字符串并写入
            f.write(metrics_df.to_string())
            f.write("\n\n")

            # 添加性能排名
            f.write("PERFORMANCE RANKING:\n")
            f.write("-" * 40 + "\n")

            # 按DPAA排名
            if "DPAA" in metrics_df.columns:
                # 移除百分号并转换为浮点数
                try:
                    dpaa_values = []
                    dpaa_methods = []
                    for method in metrics_df.index:
                        val_str = str(metrics_df.loc[method, "DPAA"])
                        # 移除可能的百分号或其他非数字字符
                        val_clean = val_str.replace('%', '').strip()
                        try:
                            dpaa_values.append(float(val_clean))
                            dpaa_methods.append(method)
                        except:
                            continue

                    if dpaa_values:
                        # 按值排序
                        sorted_indices = np.argsort(dpaa_values)[::-1]  # 降序
                        f.write("Rank by DPAA (higher is better):\n")
                        for i, idx in enumerate(sorted_indices, 1):
                            method = dpaa_methods[idx]
                            value = metrics_df.loc[method, "DPAA"]
                            f.write(f"  {i}. {method}: {value}\n")
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error in DPAA ranking: {e}\n")

            # 按Recov. Steps排名
            if "Recov. Steps" in metrics_df.columns:
                try:
                    recovery_values = []
                    recovery_methods = []

                    for method in metrics_df.index:
                        val = metrics_df.loc[method, "Recov. Steps"]
                        if isinstance(val, str):
                            if val == ">Max":
                                recovery_values.append(float('inf'))
                            else:
                                try:
                                    recovery_values.append(float(val))
                                except:
                                    recovery_values.append(float('inf'))
                        else:
                            recovery_values.append(float(val) if not np.isnan(val) else float('inf'))
                        recovery_methods.append(method)

                    # 按值排序（升序）
                    sorted_indices = np.argsort(recovery_values)
                    f.write("Rank by Recovery Steps (lower is better):\n")
                    for i, idx in enumerate(sorted_indices, 1):
                        method = recovery_methods[idx]
                        value = metrics_df.loc[method, "Recov. Steps"]
                        f.write(f"  {i}. {method}: {value}\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Error in Recovery Steps ranking: {e}\n")

        print(f"[Success] Summary report saved to: {filepath}")
        return filepath


if __name__ == "__main__":
    # 配置
    N_SAMPLES = 3000
    DRIFT_START = 1000
    DELAY = 500
    WINDOW = 100  # 小窗口以突显变化

    # 模拟真实标签 (不重要，主要看 errors)
    y_true = np.ones(N_SAMPLES)

    # --- 模拟 Wait-and-See (被动等待) ---
    # 漂移前: 错误率低 (0.1)
    # 延迟期: 错误率极高 (0.8) -> 模拟"瞎了"
    # 延迟后: 缓慢恢复 (0.2)
    errors_base = np.concatenate([
        np.random.choice([0, 1], DRIFT_START, p=[0.9, 0.1]),
        np.random.choice([0, 1], DELAY, p=[0.2, 0.8]),
        np.random.choice([0, 1], N_SAMPLES - DRIFT_START - DELAY, p=[0.8, 0.2])
    ])

    # --- 模拟 Ours (主动适应) ---
    # 漂移前: 错误率低 (0.1)
    # 延迟期: 错误率略有升高但依然不错 (0.2) -> 模拟利用伪标签适应
    # 延迟后: 恢复优秀 (0.1)
    errors_ours = np.concatenate([
        np.random.choice([0, 1], DRIFT_START, p=[0.9, 0.1]),
        np.random.choice([0, 1], DELAY, p=[0.8, 0.2]),
        np.random.choice([0, 1], N_SAMPLES - DRIFT_START - DELAY, p=[0.9, 0.1])
    ])

    results = {
        'Wait-and-See': {'errors': errors_base, 'preds': []},
        'Ours': {'errors': errors_ours, 'preds': []}
    }

    # 初始化评估器
    evaluator = DriftEvaluator(
        y_true=y_true,
        results_dict=results,
        drift_start=DRIFT_START,
        delay_rounds=DELAY,
        dataset="Test_Simulation",
        window_size=WINDOW
    )

    # 1. 打印指标表
    print("\n=== Performance Metrics ===")
    metrics_df = evaluator.get_metrics_table()
    print(metrics_df)

    # 2. 绘制曲线并保存
    print("\n=== Plotting Curves ===")

    # 使用与实际实验相同的目录结构
    # 实际实验保存目录: ../Result/baseline/{dataset}
    # 模拟测试保存到: ../Result/baseline/Test_Simulation/
    simulation_dataset = "Test_Simulation"
    simulation_save_dir = "../Result/baseline/" + simulation_dataset

    # 确保目录存在
    os.makedirs(simulation_save_dir, exist_ok=True)

    # 绘制并保存图表
    evaluator.plot_curves(
        title=f"Simulation: Impact of Label Delay ({DELAY}) on Concept Drift Adaptation",
        save_dir=simulation_save_dir
    )

    # 3. 保存结果记录 - 使用与实际实验相同的ResultRecorder
    # 注意：需要从适当的位置导入ResultRecorder
    try:
        # 尝试从不同的可能位置导入ResultRecorder
        from result_recorder import ResultRecorder
    except ImportError:
        try:
            from tool.result_recorder import ResultRecorder
        except ImportError:
            try:
                from evaluation.result_recorder import ResultRecorder
            except ImportError:
                # 如果找不到，定义一个简单的版本
                class ResultRecorder:
                    def __init__(self, base_save_dir):
                        self.base_save_dir = base_save_dir
                        os.makedirs(base_save_dir, exist_ok=True)

                    def save_metrics_table(self, metrics_df, dataset, drift_start, delay_rounds,
                                           window_size, batch_size=None, window_param=None,
                                           file_format="csv"):
                        filename = f"{dataset}_Drift{drift_start}_Delay{delay_rounds}_Win{window_size}_metrics.csv"
                        filepath = os.path.join(self.base_save_dir, filename)
                        metrics_df.to_csv(filepath)
                        print(f"[Simulation] Metrics saved to: {filepath}")
                        return filepath

                    def save_detailed_results(self, results_dict, y_true, dataset, drift_start,
                                              delay_rounds, window_size, batch_size=None,
                                              window_param=None, save_probs=True):
                        # 简化实现
                        print("[Simulation] Detailed results saving skipped in simple mode")
                        return None

    # 初始化结果记录器 - 使用与实际实验相同的参数
    recorder = ResultRecorder(base_save_dir=simulation_save_dir)

    # 保存指标表格
    metrics_csv_path = recorder.save_metrics_table(
        metrics_df=metrics_df,
        dataset=simulation_dataset,
        drift_start=DRIFT_START,
        delay_rounds=DELAY,
        window_size=WINDOW,
        file_format="csv"
    )

    # 保存详细结果
    detailed_csv_path = recorder.save_detailed_results(
        results_dict=results,
        y_true=y_true,
        dataset=simulation_dataset,
        drift_start=DRIFT_START,
        delay_rounds=DELAY,
        window_size=WINDOW,
        save_probs=False  # 模拟测试中没有概率值
    )

    # 保存汇总报告（如果方法存在）
    if hasattr(recorder, 'save_summary_report'):
        summary_path = recorder.save_summary_report(
            metrics_df=metrics_df,
            dataset=simulation_dataset,
            drift_start=DRIFT_START,
            delay_rounds=DELAY,
            window_size=WINDOW
        )

    print("\n=== Simulation Complete ===")
    print(f"Results saved to: {simulation_save_dir}")
    print(f"Metrics saved to: {metrics_csv_path}")
    if detailed_csv_path:
        print(f"Detailed results saved to: {detailed_csv_path}")

# 在代码开头添加导入
import os
import json
import pandas as pd


# 在导入语句之后，main函数之前添加一个辅助函数
def save_final_results(results, dataset, drift_start, delay_rounds, window_size,
                       batch_size, window_param, base_save_dir="../Result/baseline/"):
    """
    保存最终结果字典到文件

    Parameters:
    -----------
    results : dict
        结果字典
    dataset : str
        数据集名称
    drift_start : int
        漂移开始位置
    delay_rounds : int
        延迟轮数
    window_size : int
        窗口大小
    batch_size : int
        批次大小
    window_param : int
        窗口参数
    base_save_dir : str
        基础保存目录
    """
    # 创建结果目录
    save_dir = os.path.join(base_save_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名
    param_suffix = f"{dataset}_Drift{drift_start}_Delay{delay_rounds}_Win{window_size}"

    # 保存为CSV
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Method'

    csv_path = os.path.join(save_dir, f"{param_suffix}_results.csv")
    results_df.to_csv(csv_path)
    print(f"Final results (CSV) saved to: {csv_path}")

    # 保存为JSON
    json_path = os.path.join(save_dir, f"{param_suffix}_results.json")

    # 将numpy类型转换为Python原生类型以便JSON序列化
    json_results = {}
    for key, value in results.items():
        json_results[key] = {}
        for k, v in value.items():
            if isinstance(v, (np.floating, np.integer)):
                json_results[key][k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                json_results[key][k] = v.tolist()
            else:
                json_results[key][k] = v

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)

    print(f"Final results (JSON) saved to: {json_path}")

    return csv_path, json_path
# --- 新增：独立的画图函数 ---
def plot_loss_curves(loss_x_history, loss_z_history, save_dir='./Results', file_name='loss_curve.png', window_size=50):
    """
    绘制 Loss 曲线并保存到指定目录

    参数:
    - loss_x_history: list, Baseline 模型 Loss 历史
    - loss_z_history: list, Imputed 模型 Loss 历史
    - save_dir: str, 图片保存的文件夹路径 (如不存在会自动创建)
    - file_name: str, 图片的文件名 (包含后缀, 如 .png)
    - window_size: int, 移动平均的窗口大小
    """

    # 1. 路径处理逻辑
    # 如果目录不存在，则自动创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")

    # 拼接完整路径 (兼容 Windows/Linux)
    full_save_path = os.path.join(save_dir, file_name)

    # 2. 绘图逻辑
    plt.figure(figsize=(12, 6))

    # 绘制原始 Loss (设置透明度 alpha=0.2 以便作为背景)
    plt.plot(loss_x_history, label='Loss X (Raw)', color='blue', alpha=0.2)
    plt.plot(loss_z_history, label='Loss Z (Raw)', color='orange', alpha=0.2)

    # 绘制移动平均线 (观察主要趋势)
    if len(loss_x_history) > window_size:
        # 使用 Pandas 计算 Rolling Mean
        loss_x_ma = pd.Series(loss_x_history).rolling(window=window_size).mean()
        loss_z_ma = pd.Series(loss_z_history).rolling(window=window_size).mean()

        plt.plot(loss_x_ma, label=f'Loss X (MA-{window_size})', color='blue', linewidth=2)
        plt.plot(loss_z_ma, label=f'Loss Z (MA-{window_size})', color='orange', linewidth=2)

    # 图表装饰
    plt.title(f'Instantaneous Training Loss over Time (Window={window_size})')
    plt.xlabel('Update Steps')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 3. 保存与清理
    try:
        plt.savefig(full_save_path, dpi=300)  # dpi=300 保证图片清晰
        print(f"Loss curve saved successfully at: {full_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()  # 务必关闭，防止内存泄漏