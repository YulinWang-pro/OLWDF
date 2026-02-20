import numpy as np
import math

import numpy as np


class FTRL_ADP:
    def __init__(self, decay, L1, L2, LP, adaptive, n_inputs):
        self.ADAPTIVE = adaptive

        self.L1 = L1
        self.L2 = L2
        self.LP = LP
        self.v = np.zeros(n_inputs)
        self.h = np.zeros(n_inputs)
        self.z = np.zeros(n_inputs)

        self.r = 1
        self.d = 1 / (self.L2 + self.LP * self.r)
        self.decay = decay

        self.times = 0
        self.fails = 0
        self.times_warn = 0
        self.fails_warn = 0
        self.p_min = 1.
        self.s_min = 10.

    def fit_soft_labels(self, idx, x, y_soft, decay_choice, contribute_error_rate,is_delayed=False, delay_compensation=1.0,weight=1.0):
        """
        专门处理软标签训练的方法
        y_soft: 概率分布 [p_neg, p_pos] 或 one-hot 编码
        """
        # 获取当前权重
        w = self.weight_update(idx)

        # 计算线性预测值 (logits)
        x_val = np.dot(w, x)
        p_pred = self.__sigmoid(x_val)

        # 处理不同类型的软标签输入
        if isinstance(y_soft, (list, np.ndarray)) and len(y_soft) == 2:
            # 二分类的概率分布 [p_neg, p_pos]
            y_neg, y_pos = y_soft
        else:
            # 单个概率值 (正类概率)
            y_neg = 1 - y_soft
            y_pos = y_soft

        # 数值稳定的软交叉熵损失
        # 使用完整概率分布信息
        loss = y_neg * max(x_val, 0) + y_pos * max(-x_val, 0) + np.log(1 + np.exp(-abs(x_val)))

        # 对于延迟标签，调整梯度补偿
        if is_delayed:
            # 延迟补偿：放大梯度以补偿延迟
            gradient_factor = delay_compensation
        else:
            gradient_factor = 1.0
        if isinstance(y_soft, (int, float)):
            return p_pred, w
        else:
            if self.ADAPTIVE:
                self.times += 1
                # 定义软标签的"失败"：最大概率类别是否与预测一致
                # 真实标签的最大概率类别
                true_max_class = 1 if y_pos >= 0.5 else 0
                # 预测的最大概率类别
                pred_max_class = 1 if p_pred >= 0.5 else 0
                # 如果最大概率类别不一致，视为失败
                self.fails += int(true_max_class != pred_max_class)
                if decay_choice == 0:
                    self.decay = (np.cbrt(self.times) - 1) / np.cbrt(self.times)
                elif decay_choice == 1:
                    self.decay = (np.sqrt(self.times) - 1) / np.sqrt(self.times)
                elif decay_choice == 2:
                    self.decay = float(self.times - 1) / self.times
                elif decay_choice == 3:
                    self.decay = float(self.times) / (self.times + 1)
                elif decay_choice == 4:
                    self.decay = 1. - np.log(self.times) / (2 * self.times)

                # 错误率监控和调整
                if self.times > 30:
                    p_i = float(self.fails) / self.times
                    s_i = np.sqrt(p_i * (1 - p_i) / self.times)
                    ps = p_i + s_i

                    self.decay = self.decay * (1.0 - contribute_error_rate) + contribute_error_rate * p_i

                    if ps < self.p_min + self.s_min:
                        self.p_min = p_i
                        self.s_min = s_i
                    if ps < self.p_min + 2 * self.s_min:
                        self.times_warn = 0
                        self.fails_warn = 0
                    else:
                        self.times_warn += 1
                        self.fails_warn += int(true_max_class != pred_max_class)

                        if ps > self.p_min + 3 * self.s_min:
                            self.times = self.times_warn
                            self.fails = self.fails_warn
                            self.p_min = 1.
                            self.s_min = 10.
            if type(x) == list:
                x = np.array(x)
            # 梯度计算 (基于完整概率分布)
            g = weight*(p_pred - y_pos) * x*gradient_factor

        # 更新参数 (跳过自适应衰减部分，因为软标签不适合硬错误率计算)
        g_norm = np.linalg.norm(g)
        if g_norm > 100.0:  # 阈值根据业务调整
            print(f"Warning: Large Gradient Norm {g_norm:.2f} detected at time {self.times}")

        self.v[idx] = self.v[idx] + g
        self.h[idx] = self.decay * self.h[idx] + w
        self.z[idx] = self.v[idx] - self.LP * self.h[idx]
        self.r = 1 + self.decay * self.r
        decay_values = self.decay

        return p_pred, decay_values, loss, w

    def fit(self, idx, x, y, decay_choice, contribute_error_rate, is_delayed=False, delay_compensation=1.0,weight=1):
        """
        处理硬标签训练的方法 (y = 0 或 1)
        """
        # 获取当前权重
        w = self.weight_update(idx)
        x_val = np.dot(w, x)
        p = self.__sigmoid(x_val)
        loss = self.__loss(y, x_val)
        # 对于延迟标签，调整梯度补偿
        if is_delayed:
            # 延迟补偿：放大梯度以补偿延迟
            gradient_factor = delay_compensation
        else:
            gradient_factor = 1.0

        # 特殊处理标签为100的情况
        if y == 100:
            return p, w
        else:
            # 自适应衰减率更新
            if self.ADAPTIVE:
                self.times += 1
                self.fails += int(np.abs(y - p) > 0.5)
                if decay_choice == 0:
                    self.decay = (np.cbrt(self.times) - 1) / np.cbrt(self.times)
                elif decay_choice == 1:
                    self.decay = (np.sqrt(self.times) - 1) / np.sqrt(self.times)
                elif decay_choice == 2:
                    self.decay = float(self.times - 1) / self.times
                elif decay_choice == 3:
                    self.decay = float(self.times) / (self.times + 1)
                elif decay_choice == 4:
                    self.decay = 1. - np.log(self.times) / (2 * self.times)

                # 错误率监控和调整
                if self.times > 30:
                    p_i = float(self.fails) / self.times
                    s_i = np.sqrt(p_i * (1 - p_i) / self.times)
                    ps = p_i + s_i

                    self.decay = self.decay * (1.0 - contribute_error_rate) + contribute_error_rate * p_i

                    if ps < self.p_min + self.s_min:
                        self.p_min = p_i
                        self.s_min = s_i
                    if ps < self.p_min + 2 * self.s_min:
                        self.times_warn = 0
                        self.fails_warn = 0
                    else:
                        self.times_warn += 1
                        self.fails_warn += int(np.abs(y - p) > 0.5)

                        if ps > self.p_min + 3 * self.s_min:
                            self.times = self.times_warn
                            self.fails = self.fails_warn
                            self.p_min = 1.
                            self.s_min = 10.

            # 更新参数
            if type(x) == list:
                x = np.array(x)
            g = weight*(p - y) * x* gradient_factor

            # g_norm = np.linalg.norm(g)
            # if g_norm > 100.0:  # 阈值根据业务调整
            #     print(f"Warning: Large Gradient Norm {g_norm:.2f} detected at time {self.times}")
            self.v[idx] = self.v[idx] + g
            self.h[idx] = self.decay * self.h[idx] + w
            self.z[idx] = self.v[idx] - self.LP * self.h[idx]
            self.r = 1 + self.decay * self.r
            decay_values = self.decay

            return p, decay_values, loss, w

    def weight_update(self, idx):
        w = np.zeros(len(idx))
        mask = np.abs(self.z[idx]) > self.L1

        z_i = self.z[idx][mask]

        tmp_1_ = z_i - self.L1 * np.sign(z_i)
        tmp_2_ = self.L2 + self.LP * self.r

        w[mask] = -np.divide(tmp_1_, tmp_2_)

        return w

    def predict(self, idx, x):
        w = self.weight_update(idx)
        x_val = np.dot(w, x)
        return self.__sigmoid(x_val)

    def __sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    def __loss(self, y, x):
        """
        硬标签的数值稳定损失函数
        """
        return max(x, 0) - y * x + np.log(1 + np.exp(-abs(x)))