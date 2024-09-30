import numpy as np
import random


class Node:
    def __init__(self, state, desired_selection_policy, cp, parent=None):
        self.cp = cp
        self.desired_selection_policy = desired_selection_policy
        self.state = state  # State is a dictionary representing the current situation
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visit_count = 0  # Number of times this node has been visited
        self.total_cost = 0  # Total cost accumulated in simulations from this node
        self.scores = []

    def add_child(self, child_state):
        child_node = Node(
            state=child_state,
            desired_selection_policy=self.desired_selection_policy,
            cp=self.cp,
            parent=self,
        )
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        # if self.parent is None:
        #    return False
        return len(self.children) > 0 and all(
            child.visit_count > 0 for child in self.children
        )

    def update(self, result):
        self.visit_count += 1
        self.total_cost += result
        self.scores.append(result)

    def UCB(self, c_param):
        epsilon = 0

        visited_children = [child for child in self.children if (child.visit_count > 0)]

        sorted_children = sorted(
            visited_children,
            key=lambda child: child.total_cost / (child.visit_count + epsilon),
        )
        scores = {child: rank + 1 for rank, child in enumerate(sorted_children)}
        total_scores = sum(scores.values())

        def normalized_score(child):
            return scores[child] / total_scores

        choices_weights = [
            normalized_score(child)
            + c_param
            * (2 * np.log(self.visit_count) / (child.visit_count + epsilon)) ** 0.5
            for child in visited_children
        ]

        best_child_node = self.children[np.argmin(choices_weights)]

        return best_child_node

    def SP(self):
        visited_children = [child for child in self.children if child.visit_count > 0]
        D = 1

        def sp_mcts_score(child):
            mean_cost = np.mean(child.scores) if len(child.scores) > 0 else 0
            variance = np.var(child.scores) if len(child.scores) > 0 else 0
            possible_deviation = np.sqrt(variance + (D / child.visit_count))
            return mean_cost - self.cp * possible_deviation

        choices_weights = [sp_mcts_score(child) for child in visited_children]

        best_child_node = self.children[np.argmin(choices_weights)]
        return best_child_node

    def Bayesian(self):
        visited_children = [child for child in self.children if child.visit_count > 0]
        N = self.visit_count

        def bayesian_uct_score(child, use_variance=False):
            mean_cost = np.mean(child.scores) if len(child.scores) > 0 else 0
            exploration_term = np.sqrt(2 * np.log(N) / child.visit_count)

            if use_variance:
                variance = np.sqrt(np.var(child.scores)) if len(child.scores) > 0 else 0
                exploration_term *= variance

            return mean_cost + exploration_term

        # Select which Bayesian UCT formula to use
        use_variance = True  # Change this to `False` to use the first formula
        choices_weights = [
            bayesian_uct_score(child, use_variance=use_variance)
            for child in visited_children
        ]

        best_child_node = self.children[np.argmin(choices_weights)]
        return best_child_node

    def UCB1_tuned(self, c_param):
        visited_children = [child for child in self.children if child.visit_count > 0]

        def ucb1_tuned_score(child):
            mean_cost = np.mean(child.scores) if len(self.scores) > 1 else 0
            variance = np.var(self.scores) if len(self.scores) > 1 else 0
            # UCB1-Tuned formula
            exploration_term = np.sqrt(
                (np.log(self.visit_count) / child.visit_count)
                * min(
                    0.25,
                    variance
                    + np.sqrt(2 * np.log(self.visit_count) / child.visit_count),
                )
            )
            return mean_cost + c_param * exploration_term

        choices_weights = [ucb1_tuned_score(child) for child in visited_children]

        best_child_node = self.children[np.argmin(choices_weights)]
        return best_child_node

    def thompson_sampling(self, c_param):
        visited_children = [child for child in self.children if child.visit_count > 0]

        def best_fit_distribution(scores):
            distributions = {
                "normal": norm,
                "beta": beta,
                "exponential": expon,
                "gamma": gamma,
                "lognormal": lognorm,
                "weibull_min": weibull_min,
                "uniform": uniform,
                "pareto": pareto,
                "t": t,
                "chi2": chi2,
            }
            p_values = {}
            for dist_name, dist in distributions.items():
                try:
                    params = dist.fit(scores)
                    d_statistic, p_value = kstest(scores, dist_name, args=params)
                    p_values[dist_name] = p_value
                except Exception as e:
                    p_values[dist_name] = (
                        0  # Handle the error and skip this distribution
                    )
                    print(f"Skipping {dist_name} due to fitting issues: {e}")

            best_dist_name = max(p_values, key=p_values.get)
            best_p_value = p_values[best_dist_name]

            if best_p_value < 0.05:
                return None, None

            best_dist = distributions[best_dist_name]
            best_params = best_dist.fit(scores)

            return best_dist, best_params

        sampled_values = []
        for child in visited_children:
            if len(child.scores) > 1:
                best_dist, best_params = best_fit_distribution(child.scores)
                if best_dist is not None:
                    sampled_value = best_dist.rvs(*best_params)
                    sampled_values.append(sampled_value)
                else:
                    return self.UCB(
                        c_param
                    )  # Fallback to UCB if no good distribution is found
            else:
                sampled_values.append(np.mean(child.scores))

        best_child_node = visited_children[np.argmin(sampled_values)]
        return best_child_node

    def randomized_ucb(self, c_param, random_factor=0.1):
        visited_children = [child for child in self.children if child.visit_count > 0]

        def randomized_ucb_score(child):
            mean_cost = np.mean(child.scores) if len(child.scores) > 0 else 0
            exploration_term = np.sqrt(
                (2 * np.log(self.visit_count) / (child.visit_count))
            )
            random_term = random_factor * np.random.rand()
            return mean_cost + c_param * exploration_term + random_term

        choices_weights = [randomized_ucb_score(child) for child in visited_children]

        best_child_node = visited_children[np.argmin(choices_weights)]
        return best_child_node

    def epsilon_greedy(self, epsilon):
        visited_children = [child for child in self.children if child.visit_count > 0]

        if np.random.rand() < epsilon:
            # Explore: randomly select a child
            best_child_node = np.random.choice(visited_children)
        else:
            # Exploit: select the child with the best average cost
            best_child_node = min(
                visited_children,
                key=lambda child: (
                    np.mean(child.scores) if len(child.scores) > 0 else float("inf")
                ),
            )

        return best_child_node

    def best_child(self):
        if self.desired_selection_policy == "UCB":
            return self.UCB(c_param=self.cp)
        if self.desired_selection_policy == "UCB1T":
            return self.UCB1_tuned(c_param=self.cp)
        if self.desired_selection_policy == "SP":
            return self.epsilon_greedy(self.cp)
        if self.desired_selection_policy == "Bayesian":
            return self.Bayesian()

        else:
            raise ValueError(
                f"Unknown Selection policy: {self.desired_selection_policy}"
            )

    def delete_node(self):
        self.parent.children = [
            child for child in self.parent.children if child != self
        ]
