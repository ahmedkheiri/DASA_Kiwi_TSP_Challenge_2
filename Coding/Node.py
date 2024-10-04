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

    def best_child(self):
        if self.desired_selection_policy == "UCB":
            return self.UCB(c_param=self.cp)
        if self.desired_selection_policy == "UCB1T":
            return self.UCB1_tuned(c_param=self.cp)
        else:
            raise ValueError(
                f"Unknown Selection policy: {self.desired_selection_policy}"
            )

    def delete_node(self):
        self.parent.children = [
            child for child in self.parent.children if child != self
        ]
