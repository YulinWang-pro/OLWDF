import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class SemiSupervisedEnhancedProcessor:
    def __init__(self, delay_steps=3, buffer_size=3, lambda_decay=0.1, num_classes=2, fusion_method="fixed",
                 view1_weight=0.5, view2_weight=0.5,
                 confidence_threshold=0.6, model="LogisticRegression"):
        self.delay_steps = delay_steps
        self.buffer_size = buffer_size
        self.lambda_decay = lambda_decay
        self.num_classes = num_classes

        if model == "LogisticRegression":
            self.self_train_clf = LogisticRegression(random_state=42, max_iter=1000)
            self.self_train_clf_view1 = LogisticRegression(random_state=42, max_iter=1000)
            self.self_train_clf_view2 = LogisticRegression(random_state=42, max_iter=1000)
        if model == "svc":
            self.self_train_clf_view1 = LinearSVC(random_state=42, tol=1e-5)
            self.self_train_clf_view2 = LinearSVC(random_state=42, tol=1e-5)
            self.self_train_clf = LinearSVC(random_state=42, tol=1e-5)
        self.model = model

        self.fusion_method = fusion_method
        self.view1_weight = view1_weight
        self.view2_weight = view2_weight
        self.confidence_threshold = confidence_threshold

        self.soft_label_matrix = np.ones((num_classes, num_classes)) / num_classes
        self.soft_label_matrix_x = np.ones((num_classes, num_classes)) / num_classes
        self.soft_label_matrix_z = np.ones((num_classes, num_classes)) / num_classes

        self.smoothing_alpha = 0.7
        self.min_confidence = 0.6

    def _confidence_based_fusion_simple(self, view1_soft, view2_soft):
        view1_confidence = np.max(view1_soft)
        view2_confidence = np.max(view2_soft)

        confidence_total = view1_confidence + view2_confidence
        if confidence_total > 0:
            weight_view1 = view1_confidence / confidence_total
            weight_view2 = view2_confidence / confidence_total
        else:
            weight_view1 = weight_view2 = 0.5

        fused_label = weight_view1 * view1_soft + weight_view2 * view2_soft
        return fused_label

    def _entropy_based_fusion_simple(self, view1_soft, view2_soft):
        view1_entropy = -np.sum(view1_soft * np.log(view1_soft + 1e-10))
        view2_entropy = -np.sum(view2_soft * np.log(view2_soft + 1e-10))

        entropy_total = view1_entropy + view2_entropy
        if entropy_total > 0:
            weight_view1 = (1 - view1_entropy / entropy_total)
            weight_view2 = (1 - view2_entropy / entropy_total)
            weight_sum = weight_view1 + weight_view2
            if weight_sum > 0:
                weight_view1 /= weight_sum
                weight_view2 /= weight_sum
            else:
                weight_view1 = weight_view2 = 0.5
        else:
            weight_view1 = weight_view2 = 0.5

        fused_label = weight_view1 * view1_soft + weight_view2 * view2_soft
        return fused_label

    def _agreement_based_fusion_simple(self, view1_soft, view2_soft):
        view1_pred_class = np.argmax(view1_soft)
        view2_pred_class = np.argmax(view2_soft)

        if view1_pred_class == view2_pred_class:
            fused_label = 0.5 * view1_soft + 0.5 * view2_soft
        else:
            view1_confidence = np.max(view1_soft)
            view2_confidence = np.max(view2_soft)

            if view1_confidence > view2_confidence:
                fused_label = view1_soft
            else:
                fused_label = view2_soft

        return fused_label

    def _fixed_weight_fusion(self, view1_soft, view2_soft):
        weight_sum = self.view1_weight + self.view2_weight
        if weight_sum > 0:
            w1 = self.view1_weight / weight_sum
            w2 = self.view2_weight / weight_sum
        else:
            w1 = w2 = 0.5

        fused_label = w1 * view1_soft + w2 * view2_soft
        return fused_label

    def _view_quality_fusion(self, view1_soft, view2_soft, classifier_view1, classifier_view2,
                             labeled_indices, all_features_view1, all_features_view2):
        view1_quality = self._evaluate_view_quality(classifier_view1, all_features_view1, labeled_indices)
        view2_quality = self._evaluate_view_quality(classifier_view2, all_features_view2, labeled_indices)

        quality_total = view1_quality + view2_quality
        if quality_total > 0:
            weight_view1 = view1_quality / quality_total
            weight_view2 = view2_quality / quality_total
        else:
            weight_view1 = weight_view2 = 0.5

        fused_label = weight_view1 * view1_soft + weight_view2 * view2_soft
        return fused_label

    def generate_child_summary_vector(self, current_idx, parent_indices, node_soft_labels, features):
        if not parent_indices:
            return None

        summary_vector = np.zeros(self.num_classes)
        total_weight = 0.0

        for parent_idx in parent_indices:
            parent_soft_label = node_soft_labels[parent_idx]

            current_feature = features[current_idx]
            parent_feature = features[parent_idx]
            distance = np.linalg.norm(current_feature - parent_feature)

            weight = np.exp(-distance)

            summary_vector += weight * parent_soft_label
            total_weight += weight

        return summary_vector / total_weight if total_weight > 0 else None

    def calibrate_with_collective_wisdom(self, current_pred, pred_class, isfuza=False, view_name=None):
        if isfuza:
            collective_experience = self.soft_label_matrix[pred_class]
            current_confidence = np.max(current_pred)

            if current_confidence < 0.7:
                alpha = 0.7
            else:
                alpha = 0.3

            calibrated_label = alpha * collective_experience + (1 - alpha) * current_pred
            return calibrated_label / np.sum(calibrated_label)
        else:
            if view_name == None:
                return self.soft_label_matrix[pred_class].copy()
            if "view1" in view_name:
                return self.soft_label_matrix_x[pred_class].copy()
            if "view2" in view_name:
                return self.soft_label_matrix_z[pred_class].copy()

    def is_correctly_classified(self, idx, true_labels, features):
        try:
            true_label = true_labels[idx]
            if true_label is None:
                return False

            feature = features[idx]
            prediction = self.self_train_clf.predict([feature])[0]
            return prediction == true_label
        except:
            return False

    def is_correctly_classified_x(self, idx, true_labels, features):
        try:
            true_label = true_labels[idx]
            if true_label is None:
                return False

            feature = features[idx]
            prediction = self.self_train_clf_view1.predict([feature])[0]
            return prediction == true_label
        except:
            return False

    def is_correctly_classified_z(self, idx, true_labels, features):
        try:
            true_label = true_labels[idx]
            if true_label is None:
                return False

            feature = features[idx]
            prediction = self.self_train_clf_view2.predict([feature])[0]
            return prediction == true_label
        except:
            return False

    def update_soft_label_matrix(self, labeled_indices, true_labels, features):
        update_count = 0
        for idx in labeled_indices:
            if self.is_correctly_classified(idx, true_labels, features):
                true_label = int(true_labels[idx])

                try:
                    feature = features[idx]

                    if self.model == "LogisticRegression":
                        pred_proba = self.self_train_clf.predict_proba([feature])[0]
                    if self.model == "svc":
                        pred_proba = self.self_train_clf._predict_proba_lr([feature])[0]

                    self.soft_label_matrix[true_label] += pred_proba
                    update_count += 1
                except Exception as e:
                    continue

        if update_count > 0:
            row_sums = self.soft_label_matrix.sum(axis=1, keepdims=True)
            self.soft_label_matrix = np.divide(self.soft_label_matrix, row_sums,
                                               out=np.ones_like(self.soft_label_matrix) / self.num_classes,
                                               where=row_sums != 0)

    def update_soft_label_matrix_x(self, labeled_indices, true_labels, features):
        update_count = 0
        for idx in labeled_indices:
            if self.is_correctly_classified_x(idx, true_labels, features):
                true_label = int(true_labels[idx])

                try:
                    feature = features[idx]
                    if self.model == "LogisticRegression":
                        pred_proba = self.self_train_clf_view1.predict_proba([feature])[0]
                    if self.model == "svc":
                        pred_proba = self.self_train_clf_view1._predict_proba_lr([feature])[0]

                    self.soft_label_matrix_x[true_label] += pred_proba
                    update_count += 1
                except Exception as e:
                    continue

        if update_count > 0:
            row_sums = self.soft_label_matrix_x.sum(axis=1, keepdims=True)
            self.soft_label_matrix_x = np.divide(self.soft_label_matrix_x, row_sums,
                                                 out=np.ones_like(self.soft_label_matrix_x) / self.num_classes,
                                                 where=row_sums != 0)

    def update_soft_label_matrix_z(self, labeled_indices, true_labels, features):
        update_count = 0
        for idx in labeled_indices:
            if self.is_correctly_classified_z(idx, true_labels, features):
                true_label = int(true_labels[idx])

                try:
                    feature = features[idx]

                    if self.model == "LogisticRegression":
                        pred_proba = self.self_train_clf_view2.predict_proba([feature])[0]
                    if self.model == "svc":
                        pred_proba = self.self_train_clf_view2._predict_proba_lr([feature])[0]

                    self.soft_label_matrix_z[true_label] += pred_proba
                    update_count += 1
                except Exception as e:
                    continue

        if update_count > 0:
            row_sums = self.soft_label_matrix_z.sum(axis=1, keepdims=True)
            self.soft_label_matrix_z = np.divide(self.soft_label_matrix_z, row_sums,
                                                 out=np.ones_like(self.soft_label_matrix_z) / self.num_classes,
                                                 where=row_sums != 0)

    def geometric_label_fusion(self, source_soft_label, target_pred, geometric_distance, direction="forward"):
        distance_weight = np.exp(-geometric_distance / 2.0)
        confidence = np.max(target_pred)
        confidence_weight = min(confidence, 0.8)

        if direction == "forward":
            alpha = 0.7 * distance_weight + 0.3 * confidence_weight
        else:
            alpha = 0.5 * distance_weight + 0.5 * confidence_weight

        alpha = np.clip(alpha, 0.2, 0.8)
        fused_label = alpha * source_soft_label + (1 - alpha) * target_pred
        fused_label = fused_label / np.sum(fused_label)

        return fused_label

    def generate_parent_summary_vector(self, current_idx, parent_indices, node_soft_labels, features):
        if not parent_indices:
            return None

        summary_vector = np.zeros(self.num_classes)
        total_weight = 0.0

        for parent_idx in parent_indices:
            parent_soft_label = node_soft_labels[parent_idx]

            current_feature = features[current_idx]
            parent_feature = features[parent_idx]
            distance = np.linalg.norm(current_feature - parent_feature)

            weight = np.exp(-distance)

            summary_vector += weight * parent_soft_label
            total_weight += weight

        return summary_vector / total_weight if total_weight > 0 else None

    def assess_parent_quality(self, parent_vector):
        consistency = np.max(parent_vector)
        entropy = -np.sum(parent_vector * np.log(parent_vector + 1e-10))
        max_entropy = np.log(self.num_classes)
        certainty = 1.0 - (entropy / max_entropy)
        quality = 0.7 * consistency + 0.3 * certainty
        return min(quality, 1.0)

    def _calculate_entropy(self, prob_dist):
        prob_dist = np.clip(prob_dist, 1e-10, 1.0)
        return -np.sum(prob_dist * np.log(prob_dist))

    def simplified_three_vector_fusion(self, self_vector, class_vector, parent_vector, temperature=2.0):
        try:
            vectors = [self_vector, class_vector]
            if parent_vector is not None:
                vectors.append(parent_vector)

            entropies = np.array([self._calculate_entropy(v) for v in vectors])
            raw_weights = np.exp(-entropies * temperature)
            weights = raw_weights / np.sum(raw_weights)

            fused_vector = np.zeros_like(self_vector)
            for i, vec in enumerate(vectors):
                fused_vector += weights[i] * vec

            fused_vector = fused_vector / np.sum(fused_vector)
            return fused_vector

        except Exception as e:
            print(f"Fusion error: {e}, fallback to average.")
            if parent_vector is not None:
                return (self_vector + class_vector + parent_vector) / 3.0
            else:
                return (self_vector + class_vector) / 2.0

    def geometric_soft_label_propagation(self, unlabeled_indices, labeled_indices, nneigh, all_features, batch_y_true):
        if len(labeled_indices) == 0 or len(unlabeled_indices) == 0:
            return {}, {}

        data = all_features[labeled_indices]
        label_data = batch_y_true[labeled_indices]

        data = np.array(data)
        label_data = np.array(label_data)

        self.self_train_clf.fit(data, label_data)
        self.update_soft_label_matrix(labeled_indices, batch_y_true, all_features)

        struct = np.array(labeled_indices)
        struct_record = struct.copy()

        node_soft_labels = {}

        for idx in labeled_indices:
            true_label = int(batch_y_true[idx])
            soft_label = np.zeros(self.num_classes)
            soft_label[true_label] = 1.0
            node_soft_labels[idx] = soft_label

        soft_pseudo_labels = {}
        hard_pseudo_labels = {}

        data_neigh = []
        iteration_count = 0
        max_iterations = 10

        while len(struct) > 0:
            iteration_count += 1

            data_neigh = []
            for i in range(len(struct)):
                current_idx = struct[i]
                next_idx = nneigh[int(current_idx)]
                if next_idx != current_idx:
                    data_neigh.append(next_idx)

            data_neigh = np.array(data_neigh)

            struct = np.setdiff1d(data_neigh, struct_record)
            length_struct = len(struct)

            if length_struct > 0:
                for j in range(length_struct):
                    struct_idx = struct[j]
                    try:
                        parent_idx = []
                        for k in range(len(nneigh)):
                            if nneigh[k] == struct_idx and k in node_soft_labels:
                                parent_idx.append(k)
                        if len(parent_idx) > 0:
                            if self.model == "LogisticRegression":
                                current_pred = self.self_train_clf.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                current_pred = self.self_train_clf._predict_proba_lr([all_features[struct_idx]])[0]

                            pred_class = np.argmax(current_pred)
                            class_pred = self.calibrate_with_collective_wisdom(current_pred, pred_class)

                            parent_vector = self.generate_parent_summary_vector(struct_idx, parent_idx,
                                                                                node_soft_labels, all_features)

                            fused_soft_label = self.simplified_three_vector_fusion(current_pred, class_pred,
                                                                                   parent_vector)
                        else:
                            print("No parent node")
                            if self.model == "LogisticRegression":
                                fused_soft_label = self.self_train_clf.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                fused_soft_label = self.self_train_clf._predict_proba_lr([all_features[struct_idx]])[0]

                        hard_label = np.argmax(fused_soft_label)

                        soft_pseudo_labels[struct_idx] = fused_soft_label
                        hard_pseudo_labels[struct_idx] = hard_label
                        node_soft_labels[struct_idx] = fused_soft_label

                        data = np.vstack([data, all_features[struct_idx]])
                        label_data = np.append(label_data, hard_label)

                    except Exception as e:
                        print(f"Forward propagation prediction failed: {e}")
                        continue

                self.self_train_clf.fit(data, label_data)

                for i in range(length_struct):
                    struct_record = np.append(struct_record, struct[i])
                self.update_soft_label_matrix(struct_record, batch_y_true, all_features)

        def find(condition):
            res = np.nonzero(condition)
            return res[0] if len(res) > 0 else np.array([])

        struct = struct_record
        iteration_count = 0
        data_neigh = []

        while len(struct) > 0:
            iteration_count += 1

            data_neigh = []
            for i in range(len(struct)):
                current_idx = struct[i]
                number_neigh = find(nneigh == current_idx)
                length_neigh = len(number_neigh)

                for j in range(length_neigh):
                    neighbor_idx = number_neigh[j]
                    if neighbor_idx != current_idx:
                        data_neigh.append(neighbor_idx)

            data_neigh = np.array(data_neigh)

            struct = np.setdiff1d(data_neigh, struct_record)
            length_struct = len(struct)

            if length_struct > 0:
                for j in range(length_struct):
                    struct_idx = struct[j]
                    try:
                        child_idx = nneigh[struct_idx] if struct_idx < len(nneigh) else None
                        if child_idx is not None and child_idx in node_soft_labels:
                            if self.model == "LogisticRegression":
                                current_pred = self.self_train_clf.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                current_pred = self.self_train_clf._predict_proba_lr([all_features[struct_idx]])[0]
                            pred_class = np.argmax(current_pred)
                            class_pred = self.calibrate_with_collective_wisdom(current_pred, pred_class)
                            parent_vector = self.generate_child_summary_vector(struct_idx, [child_idx],
                                                                               node_soft_labels, all_features)
                            fused_soft_label = self.simplified_three_vector_fusion(current_pred, class_pred,
                                                                                   parent_vector)
                        else:
                            print("No child node")
                            if self.model == "LogisticRegression":
                                fused_soft_label = self.self_train_clf.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                fused_soft_label = self.self_train_clf._predict_proba_lr([all_features[struct_idx]])[0]

                        hard_label = np.argmax(fused_soft_label)

                        soft_pseudo_labels[struct_idx] = fused_soft_label
                        hard_pseudo_labels[struct_idx] = hard_label
                        node_soft_labels[struct_idx] = fused_soft_label

                        data = np.vstack([data, all_features[struct_idx]])
                        label_data = np.append(label_data, hard_label)

                    except Exception as e:
                        print(f"Backward propagation prediction failed: {e}")
                        continue

                self.self_train_clf.fit(data, label_data)

                struct_record_list = list(struct_record)
                for i in range(length_struct):
                    struct_idx = struct[i]
                    struct_record_list.append(struct_idx)
                struct_record = np.array(struct_record_list)
                self.update_soft_label_matrix(struct_record, batch_y_true, all_features)

        if self.model == "LogisticRegression":
            soft_pseudo_labels = self.self_train_clf.predict_proba(all_features)
        if self.model == "svc":
            soft_pseudo_labels = self.self_train_clf._predict_proba_lr(all_features)
        hard_pseudo_labels = np.argmax(soft_pseudo_labels, axis=1)

        return hard_pseudo_labels, soft_pseudo_labels

    def _single_view_complete_propagation(self, struct, struct_record, nneigh, all_features,
                                          node_soft_labels, soft_pseudo_labels, hard_pseudo_labels,
                                          data, label_data, classifier, batch_y_true, view_name):
        max_iterations = 1000

        struct, struct_record, data, label_data = self._single_view_forward_propagation_independent(
            struct, struct_record, nneigh, all_features,
            node_soft_labels, soft_pseudo_labels, hard_pseudo_labels,
            data, label_data, classifier, max_iterations, batch_y_true, f"{view_name}_forward"
        )

        def find(condition):
            res = np.nonzero(condition)
            return res[0] if len(res) > 0 else np.array([])

        struct_backward = struct_record.copy()
        hard_labels, soft_labels = self._single_view_backward_propagation_independent(
            struct_backward, struct_record, nneigh, all_features,
            node_soft_labels, soft_pseudo_labels, hard_pseudo_labels,
            data, label_data, classifier, max_iterations, find, batch_y_true, f"{view_name}_backward"
        )

        return hard_labels, soft_labels

    def _ensemble_fusion_strategy(self, soft_pseudo_labels_view1, hard_pseudo_labels_view1,
                                  soft_pseudo_labels_view2, hard_pseudo_labels_view2,
                                  all_features_view1, all_features_view2,
                                  classifier_view1, classifier_view2, labeled_indices):
        fused_soft_pseudo_labels = {}
        fused_hard_pseudo_labels = {}

        all_nodes = set(list(soft_pseudo_labels_view1.keys()) + list(soft_pseudo_labels_view2.keys()))

        for node_idx in all_nodes:
            view1_soft = soft_pseudo_labels_view1.get(node_idx, None)
            view2_soft = soft_pseudo_labels_view2.get(node_idx, None)

            if view1_soft is not None and view2_soft is not None:
                if self.fusion_method == "confidence":
                    fused_soft_label = self._confidence_based_fusion_simple(view1_soft, view2_soft)
                elif self.fusion_method == "entropy":
                    fused_soft_label = self._entropy_based_fusion_simple(view1_soft, view2_soft)
                elif self.fusion_method == "agreement":
                    fused_soft_label = self._agreement_based_fusion_simple(view1_soft, view2_soft)
                elif self.fusion_method == "fixed":
                    fused_soft_label = self._fixed_weight_fusion(view1_soft, view2_soft)
                elif self.fusion_method == "view_quality":
                    fused_soft_label = self._view_quality_fusion(
                        view1_soft, view2_soft, classifier_view1, classifier_view2,
                        labeled_indices, all_features_view1, all_features_view2
                    )
                else:
                    fused_soft_label = self._fixed_weight_fusion(view1_soft, view2_soft)

            elif view1_soft is not None:
                fused_soft_label = view1_soft
            elif view2_soft is not None:
                fused_soft_label = view2_soft
            else:
                continue

            fused_soft_label = fused_soft_label / np.sum(fused_soft_label)

            hard_label = np.zeros(2)
            hard_label[int(np.argmax(fused_soft_label))] = 1.0

            fused_soft_pseudo_labels[node_idx] = fused_soft_label
            fused_hard_pseudo_labels[node_idx] = hard_label

        return fused_soft_pseudo_labels, fused_hard_pseudo_labels

    def _confidence_based_fusion(self, view1_soft, view2_soft, feature_view1, feature_view2,
                                 classifier_view1, classifier_view2):
        view1_confidence = np.max(view1_soft)
        view2_confidence = np.max(view2_soft)

        view1_entropy = -np.sum(view1_soft * np.log(view1_soft + 1e-10))
        view2_entropy = -np.sum(view2_soft * np.log(view2_soft + 1e-10))

        confidence_total = view1_confidence + view2_confidence
        if confidence_total > 0:
            weight_view1_conf = view1_confidence / confidence_total
            weight_view2_conf = view2_confidence / confidence_total
        else:
            weight_view1_conf = weight_view2_conf = 0.5

        entropy_total = view1_entropy + view2_entropy
        if entropy_total > 0:
            weight_view1_ent = (1 - view1_entropy / entropy_total)
            weight_view2_ent = (1 - view2_entropy / entropy_total)
            ent_sum = weight_view1_ent + weight_view2_ent
            if ent_sum > 0:
                weight_view1_ent /= ent_sum
                weight_view2_ent /= ent_sum
            else:
                weight_view1_ent = weight_view2_ent = 0.5
        else:
            weight_view1_ent = weight_view2_ent = 0.5

        view1_pred_class = np.argmax(view1_soft)
        view2_pred_class = np.argmax(view2_soft)

        if view1_pred_class == view2_pred_class:
            agreement_bonus = 0.2
        else:
            agreement_bonus = -0.1

        alpha = 0.6
        beta = 0.3
        gamma = 0.1

        weight_view1 = (alpha * weight_view1_conf +
                        beta * weight_view1_ent +
                        gamma * (0.5 + agreement_bonus))

        weight_view2 = (alpha * weight_view2_conf +
                        beta * weight_view2_ent +
                        gamma * (0.5 - agreement_bonus))

        weight_view1 = np.clip(weight_view1, 0.1, 0.9)
        weight_view2 = np.clip(weight_view2, 0.1, 0.9)

        weight_sum = weight_view1 + weight_view2
        if weight_sum > 0:
            weight_view1 /= weight_sum
            weight_view2 /= weight_sum
        else:
            weight_view1 = weight_view2 = 0.5

        fused_soft_label = 0.5 * view1_soft + 0.5 * view2_soft

        return fused_soft_label

    def _weighted_fusion_based_on_view_quality(self, view1_soft, view2_soft,
                                               classifier_view1, classifier_view2,
                                               labeled_indices, all_features_view1, all_features_view2):
        view1_accuracy = self._evaluate_view_quality(classifier_view1, all_features_view1, labeled_indices)
        view2_accuracy = self._evaluate_view_quality(classifier_view2, all_features_view2, labeled_indices)

        total_accuracy = view1_accuracy + view2_accuracy
        if total_accuracy > 0:
            weight_view1 = view1_accuracy / total_accuracy
            weight_view2 = view2_accuracy / total_accuracy
        else:
            weight_view1 = weight_view2 = 0.5

        fused_soft_label = weight_view1 * view1_soft + weight_view2 * view2_soft

        return fused_soft_label

    def _evaluate_view_quality(self, classifier, features, labeled_indices):
        if len(labeled_indices) == 0:
            return 0.5

        correct_count = 0
        total_count = 0

        for idx in labeled_indices:
            try:
                feature = features[idx]
                true_label_idx = idx + self.delay_steps

                if 0 <= true_label_idx < len(self.buffer['true_labels']):
                    true_label = self.buffer['true_labels'][true_label_idx]

                    if true_label is not None:
                        pred = classifier.predict([feature])[0]
                        if pred == true_label:
                            correct_count += 1
                        total_count += 1
            except:
                continue

        if total_count > 0:
            accuracy = correct_count / total_count
        else:
            accuracy = 0.5

        return accuracy

    def _single_view_forward_propagation_independent(self, struct, struct_record, nneigh, all_features,
                                                     node_soft_labels, soft_pseudo_labels, hard_pseudo_labels,
                                                     data, label_data, classifier, max_iterations, batch_y_true,
                                                     view_name):
        iteration_count = 0

        while len(struct) > 0 and iteration_count < max_iterations:
            iteration_count += 1

            data_neigh = []
            for i in range(len(struct)):
                current_idx = struct[i]
                next_idx = nneigh[int(current_idx)]
                if next_idx != current_idx:
                    data_neigh.append(next_idx)

            data_neigh = np.array(data_neigh)
            struct = np.setdiff1d(data_neigh, struct_record)
            length_struct = len(struct)

            if length_struct > 0:
                for j in range(length_struct):
                    struct_idx = struct[j]
                    try:
                        parent_idx = []
                        for k in range(len(nneigh)):
                            if nneigh[k] == struct_idx and k in node_soft_labels:
                                parent_idx.append(k)
                        if len(parent_idx) > 0:
                            if self.model == "LogisticRegression":
                                current_pred = classifier.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                current_pred = classifier._predict_proba_lr([all_features[struct_idx]])[0]

                            pred_class = np.argmax(current_pred)
                            class_pred = self.calibrate_with_collective_wisdom(current_pred, pred_class,
                                                                               view_name=view_name)

                            parent_vector = self.generate_parent_summary_vector(struct_idx, parent_idx,
                                                                                node_soft_labels, all_features)

                            fused_soft_label = self.simplified_three_vector_fusion(current_pred, class_pred,
                                                                                   parent_vector)
                        else:
                            print("No parent node")
                            if self.model == "LogisticRegression":
                                fused_soft_label = self.self_train_clf.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                fused_soft_label = self.self_train_clf._predict_proba_lr([all_features[struct_idx]])[0]

                        hard_label = np.argmax(fused_soft_label)

                        soft_pseudo_labels[struct_idx] = fused_soft_label
                        hard_pseudo_labels[struct_idx] = hard_label
                        node_soft_labels[struct_idx] = fused_soft_label

                        data = np.vstack([data, all_features[struct_idx]])
                        label_data = np.append(label_data, hard_label)

                    except Exception as e:
                        print(f"{view_name} forward propagation prediction failed: {e}")
                        continue

                training_success = classifier.fit(data, label_data)
                if not training_success:
                    break

                for i in range(length_struct):
                    struct_record = np.append(struct_record, struct[i])
                self.update_soft_label_matrix(struct_record, batch_y_true, all_features)

        return struct, struct_record, data, label_data

    def _single_view_backward_propagation_independent(self, struct, struct_record, nneigh, all_features,
                                                      node_soft_labels, soft_pseudo_labels, hard_pseudo_labels,
                                                      data, label_data, classifier, max_iterations, find_func,
                                                      batch_y_true,
                                                      view_name):
        iteration_count = 0

        while len(struct) > 0 and iteration_count < max_iterations:
            iteration_count += 1

            data_neigh = []
            for i in range(len(struct)):
                current_idx = struct[i]
                number_neigh = find_func(nneigh == current_idx)
                length_neigh = len(number_neigh)

                for j in range(length_neigh):
                    neighbor_idx = number_neigh[j]
                    if neighbor_idx != current_idx:
                        data_neigh.append(neighbor_idx)

            data_neigh = np.array(data_neigh)
            struct = np.setdiff1d(data_neigh, struct_record)
            length_struct = len(struct)

            if length_struct > 0:
                for j in range(length_struct):
                    struct_idx = struct[j]
                    try:
                        child_idx = nneigh[struct_idx] if struct_idx < len(nneigh) else None
                        if child_idx is not None and child_idx in node_soft_labels:
                            if self.model == "LogisticRegression":
                                current_pred = classifier.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                current_pred = classifier._predict_proba_lr([all_features[struct_idx]])[0]
                            pred_class = np.argmax(current_pred)
                            class_pred = self.calibrate_with_collective_wisdom(current_pred, pred_class,
                                                                               view_name=view_name)
                            parent_vector = self.generate_child_summary_vector(struct_idx, [child_idx],
                                                                               node_soft_labels, all_features)
                            fused_soft_label = self.simplified_three_vector_fusion(current_pred, class_pred,
                                                                                   parent_vector)
                        else:
                            print("No child node")
                            if self.model == "LogisticRegression":
                                fused_soft_label = self.self_train_clf.predict_proba([all_features[struct_idx]])[0]
                            if self.model == "svc":
                                fused_soft_label = self.self_train_clf._predict_proba_lr([all_features[struct_idx]])[0]

                        hard_label = np.argmax(fused_soft_label)

                        soft_pseudo_labels[struct_idx] = fused_soft_label
                        hard_pseudo_labels[struct_idx] = hard_label
                        node_soft_labels[struct_idx] = fused_soft_label

                        data = np.vstack([data, all_features[struct_idx]])
                        label_data = np.append(label_data, hard_label)

                    except Exception as e:
                        print(f"{view_name} backward propagation prediction failed: {e}")
                        continue

                training_success = classifier.fit(data, label_data)
                if not training_success:
                    break

                struct_record_list = list(struct_record)
                for i in range(length_struct):
                    struct_idx = struct[i]
                    struct_record_list.append(struct_idx)
                struct_record = np.array(struct_record_list)
                self.update_soft_label_matrix(struct_record, batch_y_true, all_features)

        if self.model == "LogisticRegression":
            soft_pseudo_labels = classifier.predict_proba(all_features)
        if self.model == "svc":
            soft_pseudo_labels = classifier._predict_proba_lr(all_features)
        hard_pseudo_labels = np.argmax(soft_pseudo_labels, axis=1)

        return hard_pseudo_labels, soft_pseudo_labels

    def geometric_soft_label_propagation_ensemble(self, unlabeled_indices, labeled_indices, nneigh_view1,
                                                  all_features_view1,
                                                  all_features_view2, nneigh_view2, batch_y_true):
        if len(labeled_indices) == 0 or len(unlabeled_indices) == 0:
            return {}, {}

        use_dual_view = (all_features_view2 is not None and nneigh_view2 is not None and
                         len(all_features_view2) == len(all_features_view1))

        data_view1 = all_features_view1[labeled_indices]
        label_data_view1 = batch_y_true[labeled_indices]

        data_view2 = all_features_view2[labeled_indices]
        label_data_view2 = batch_y_true[labeled_indices]

        data_view1 = np.array(data_view1)
        label_data_view1 = np.array(label_data_view1)

        if use_dual_view:
            data_view2 = np.array(data_view2)
            label_data_view2 = np.array(label_data_view2)

        training_success_view1 = self.self_train_clf_view1.fit(data_view1, label_data_view1)
        if not training_success_view1:
            return {}, {}

        if use_dual_view:
            training_success_view2 = self.self_train_clf_view2.fit(data_view2, label_data_view2)
            if not training_success_view2:
                use_dual_view = False

        struct_view1 = np.array(labeled_indices)
        struct_record_view1 = struct_view1.copy()

        node_soft_labels_view1 = {}
        node_soft_labels_view2 = {} if use_dual_view else None

        self.update_soft_label_matrix_x(struct_view1, batch_y_true, data_view1)
        self.update_soft_label_matrix_z(struct_view1, batch_y_true, data_view2)

        for idx in labeled_indices:
            true_label = int(batch_y_true[idx])
            soft_label = np.zeros(self.num_classes)
            soft_label[true_label] = 1.0
            node_soft_labels_view1[idx] = soft_label
            if use_dual_view:
                node_soft_labels_view2[idx] = soft_label.copy()

        tmp_soft_pseudo_labels_view1 = {}
        tmp_hard_pseudo_labels_view1 = {}
        tmp_soft_pseudo_labels_view2 = {}
        tmp_hard_pseudo_labels_view2 = {}

        hard_labels_view1, soft_labels_view1 = self._single_view_complete_propagation(
            struct_view1, struct_record_view1, nneigh_view1, all_features_view1,
            node_soft_labels_view1, tmp_soft_pseudo_labels_view1, tmp_hard_pseudo_labels_view1,
            data_view1, label_data_view1, self.self_train_clf_view1, batch_y_true, "view1"
        )

        if use_dual_view:
            struct_view2 = np.array(labeled_indices)
            struct_record_view2 = struct_view2.copy()

            hard_labels_view2, soft_labels_view2 = self._single_view_complete_propagation(
                struct_view2, struct_record_view2, nneigh_view2, all_features_view2,
                node_soft_labels_view2, tmp_soft_pseudo_labels_view2, tmp_hard_pseudo_labels_view2,
                data_view2, label_data_view2, self.self_train_clf_view2, batch_y_true, "view2"
            )

        soft_labels, hard_labels = self._ensemble_fusion_strategy(
            {i: value for i, value in enumerate(soft_labels_view1)},
            {i: value for i, value in enumerate(hard_labels_view1)},
            {i: value for i, value in enumerate(soft_labels_view2)},
            {i: value for i, value in enumerate(hard_labels_view2)},
            all_features_view1, all_features_view2,
            self.self_train_clf_view1, self.self_train_clf_view2, labeled_indices
        )
        hard_p_labels = np.array([hard_labels[key] for key in sorted(hard_labels.keys())])
        soft_p_labels = np.array([soft_labels[key] for key in sorted(soft_labels.keys())])
        return hard_p_labels, soft_p_labels