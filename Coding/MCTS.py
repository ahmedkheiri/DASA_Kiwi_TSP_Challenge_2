import numpy as np
import random
from copy import deepcopy
import logging
import time
import os
import shutil
import glob

from Data_Preprocessing import data_preprocessing
from Node import Node


class MCTS(data_preprocessing):
    def __init__(
        self,
        instance,
        instance_number,
        number_childrens,
        desired_expansion_policy,
        ratio_expansion,
        desired_simulation_policy,
        desired_selection_policy,
        cp,
        number_simulation,
    ):
        self.instance_number = instance_number
        self.number_childrens = number_childrens
        self.desired_simulation_policy = desired_simulation_policy
        self.desired_expansion_policy = desired_expansion_policy
        self.ratio_expansion = ratio_expansion
        self.number_simulation = number_simulation
        self.desired_selection_policy = desired_selection_policy
        self.cp = cp

        self.expanded_nodes = []
        self.simulations_dict = {}

        self.start_time = time.time()
        super().__init__(instance_path=instance)
        self.end_time_data_preprocessing = time.time() - self.start_time
        self.simulation()
        # self.organise_log_files_in_folder(
        #    folder_path=os.path.dirname(self.instance_path)
        # )
        # self.collect_all_nodes()

    def configure_logging(self):
        log_file = f"{self.instance_path}_{self.number_childrens}_{self.desired_simulation_policy}_{self.desired_expansion_policy}_{self.ratio_expansion}_{self.number_simulation}_{self.desired_selection_policy}_{self.cp}.log"
        log_file = self.get_unique_log_file(log_file)

        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure the logger
        logging.basicConfig(
            level=logging.DEBUG,  # Set the log level to DEBUG to capture all types of logs
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    log_file, mode="w"
                ),  # 'w' to overwrite the log file each run, 'a' to append
                # logging.StreamHandler(),  # Optional: to also print logs to the console
            ],
        )
        logger = logging.getLogger(__name__)
        return logger

    def get_unique_log_file(self, base_log_file):
        """
        Check if the log file exists and if so, create a new file with a unique suffix.
        """
        base_name, extension = os.path.splitext(base_log_file)
        counter = 0  # Start with 0 to have the first file as _0
        while True:
            new_log_file = f"{base_name}_{counter}{extension}"
            if not os.path.exists(new_log_file):
                return new_log_file
            counter += 1

    def organise_log_files_in_folder(self, folder_path):
        """
        Organize log files in the specified folder by moving files with the same base name into a dedicated directory.

        :param folder_path: Path to the folder containing the log files.
        """
        # Change to the target directory
        os.chdir(folder_path)

        # Find all log files in the directory
        log_files = glob.glob("*.log")

        # Track which files have already been moved to avoid duplication
        processed_bases = set()

        for log_file in log_files:
            # Extract the base name (up to the first '_')
            base_name = (
                log_file.rsplit("_", 1)[0]
                if "_" in log_file
                else log_file.rsplit(".", 1)[0]
            )

            if base_name not in processed_bases:
                # Mark this base as processed
                processed_bases.add(base_name)

                # Create a pattern to match all similar files
                pattern = f"{base_name}*.log"

                # Find all files matching this pattern
                matching_files = glob.glob(pattern)

                if matching_files:
                    # Create a directory for these files
                    folder_name = os.path.join(folder_path, base_name)
                    os.makedirs(folder_name, exist_ok=True)

                    # Move each matching file into the directory
                    for file in matching_files:
                        shutil.move(file, folder_name)

                    # print(
                    #    f"Moved files with base '{base_name}' into folder: {folder_name}"
                    # )

    def initialise_root_node(self):
        return {
            "current_day": 1,
            "current_airport": self.starting_airport,
            "remaining_zones": [
                x for x in self.list_areas if x != self.starting_area
            ],  # Exclude the starting area
            "visited_zones": [self.starting_area],  # Exclude the starting area
            "total_cost": 0,
            "path": [self.starting_airport],
        }

    def transition_function(self, state, action):
        new_state = deepcopy(state)
        new_state["current_day"] += 1
        new_state["current_airport"] = action[0]
        new_state["total_cost"] += action[1]
        new_state["path"].append(action[0])
        # self.logger.info(
        #    f"Airport {action[0]}, {self.associated_area_to_airport(airport=action[0])} to remove in {new_state['remaining_zones']}"
        # )
        new_state["remaining_zones"].remove(
            self.associated_area_to_airport(airport=action[0])
        )
        new_state["visited_zones"].append(
            self.associated_area_to_airport(airport=action[0])
        )
        return new_state

    def random_policy(self, actions):

        if not actions:
            return None
        return random.choice(actions)

    def greedy_policy(self, actions):

        # self.logger.info(f"Actions: {actions}")
        if not actions:
            return None
        # Select the action with the lowest cost
        best_action = min(actions, key=lambda x: x[1])
        # self.logger.info(f"Chosen action based on heuristic policy: {best_action}")
        return best_action

    def tolerance_heuristic_policy(self, actions):
        # self.logger.info(f"Actions: {actions}")

        if not actions:
            return None

        # Find the minimum cost
        min_cost = min(actions, key=lambda x: x[1])[1]

        # Filter actions within the tolerance level
        best_actions = [
            action
            for action in actions
            if action[1] <= min_cost * (1 + self.ratio_expansion)
        ]

        # Select a random action from the best actions
        best_action = random.choice(best_actions)

        # self.logger.info(f"Chosen action based on tolerance policy: {best_action}")

        return best_action

    def get_unvisited_children(self, node):
        queue = [node]
        unvisited_children = []
        while queue:
            current_node = queue.pop(0)
            for child in current_node.children:
                if child.visit_count == 0:
                    unvisited_children.append(child)
                else:
                    queue.append(child)

        return unvisited_children

    def backpropagate(self, node, cost):
        while node is not None:

            node.update(cost)

            # self.logger.info(
            #    f"Backpropagating Node: {node.state}, Visit Count: {node.visit_count}, Total Cost: {node.total_cost}, Scores: {node.scores}"
            # )

            node = node.parent

    def collect_all_nodes(self):
        nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def get_final_nodes(self):
        day = self.number_of_areas + 1
        nodes = [
            node
            for node in self.collect_all_nodes()
            if node.state.get("current_day") == day
        ]

        # Initialize variables to track the best nodes for this day
        min_cost_child = None
        robust_child = None
        min_cost_robust_child = None
        secure_child = None

        # Values to compare against
        min_cost = float("inf")
        max_visit_count = -float("inf")
        max_secure_value = -float("inf")

        for node in nodes:
            # Min-Cost Child: Select the root child with the lowest total_cost
            if node.total_cost < min_cost:
                min_cost = node.total_cost
                min_cost_child = node

            # Robust Child: Select the most visited root child (visit_count)
            if node.visit_count > max_visit_count:
                max_visit_count = node.visit_count
                robust_child = node

            # Min-Cost-Robust Child: Among the nodes with the lowest total_cost, select the one with the highest visit_count
            if node.total_cost == min_cost and node.visit_count >= max_visit_count:
                min_cost_robust_child = node

            # Secure Child: Select the child that minimizes a lower confidence bound
            if (
                node.visit_count > 0
                and node.parent is not None
                and node.parent.visit_count > 0
            ):
                secure_value = (node.total_cost / node.visit_count) - self.cp * (
                    (node.parent.visit_count / node.visit_count) ** 0.5
                )
                if secure_value > max_secure_value:
                    max_secure_value = secure_value
                    secure_child = node

        # Logging the results for the current day
        if min_cost_child:
            self.logger.info("\n\n")
            self.logger.info(f"Best Node: {min_cost_child.state}")
        if robust_child:
            self.logger.info(
                f"Robust Child (Day {day}): State={robust_child.state}, Visit Count={max_visit_count}"
            )
        if min_cost_robust_child:
            self.logger.info(
                f"Min-Cost-Robust Child (Day {day}): State={min_cost_robust_child.state}, Cost={min_cost}, Visit Count={min_cost_robust_child.visit_count}"
            )
        if secure_child:
            self.logger.info(
                f"Secure Child (Day {day}): State={secure_child.state}, Secure Value={max_secure_value}"
            )

    def display_all_nodes(self, nodes):
        for node in nodes:
            print(
                f"State: {node.state}, Visit Count: {node.visit_count}, Total Cost: {node.total_cost}"
            )
            self.logger.info(
                f"State: {node.state}, Visit Count: {node.visit_count}, Total Cost: {node.total_cost}"
            )

    def print_execution_times(self):
        self.logger.info(
            f"\n\n\n Time to preprocess the data: {self.end_time_data_preprocessing:.4f} seconds"
        )
        self.logger.info(
            f"\n\n\n Time to find the solution: {self.end_search_time:.4f} seconds"
        )
        self.logger.info(
            f"\n\n\n Total time: {self.end_time_data_preprocessing+self.end_search_time:.4f} seconds \n\n"
        )

    def get_simulation_policy(self):
        if self.desired_simulation_policy == "greedy_policy":
            return self.greedy_policy
        elif self.desired_simulation_policy == "random_policy":
            return self.random_policy
        elif self.desired_simulation_policy == "tolerance_policy":
            return self.tolerance_heuristic_policy
        else:
            raise ValueError(
                f"Unknown simulation policy: {self.desired_simulation_policy}"
            )

    def get_expansion_policy(self):
        if self.desired_expansion_policy == "top_k":
            return self.top_k_actions

        if self.desired_expansion_policy == "ratio_k":
            return self.ratio_best_random

        else:
            raise ValueError(
                f"Unknown expansion policy: {self.desired_expansion_policy}"
            )

    def top_k_actions(self, actions):
        sorted_actions = sorted(actions, key=lambda x: x[1])
        return sorted_actions[: self.number_childrens]

    def ratio_best_random(self, actions):
        # Determine the number of best actions to take based on the ratio
        ratio = self.ratio_expansion
        num_best = int(self.number_childrens * ratio)
        num_random = self.number_childrens - num_best

        # Sort actions to get the best ones
        sorted_actions = sorted(actions, key=lambda x: x[1])
        best_actions = sorted_actions[:num_best]

        # Select the remaining random actions from the remaining pool
        remaining_actions = sorted_actions[num_best:]

        # Ensure we don't try to sample more than available actions
        num_random = min(num_random, len(remaining_actions))

        # If num_random is zero or there are no remaining actions, we skip the sampling
        if num_random > 0 and remaining_actions:
            random_actions = random.sample(remaining_actions, num_random)
        else:
            random_actions = []

        # Combine the best actions and the random actions
        final_actions = best_actions + random_actions
        random.shuffle(final_actions)

        return final_actions

    def delete_node(self, node):
        if node.parent:
            for _ in node.parent.children:
                pass
                # self.logger.info(
                #    f"before deletion: {len(node.parent.children)},{_.state}"
                # )
            node.parent.children.remove(node)
            for _ in node.parent.children:
                pass
                # self.logger.info(
                #    f"after deletion: {len(node.parent.children)},{_.state}"
                # )

    def print_characteristics_simulation(self):
        self.logger.info(f"\n\nSimulation dictionnary: {self.simulations_dict}")
        self.logger.info(f"Number of childrens: {self.number_childrens}")
        self.logger.info(f"Desired expansion policy: {self.desired_expansion_policy}")
        self.logger.info(f"Ratio expansion: {self.ratio_expansion}")
        self.logger.info(f"Desired simulation policy: {self.desired_simulation_policy}")
        self.logger.info(f"Desired selection policy: {self.desired_selection_policy}")
        self.logger.info(f"Cp: {self.cp}")
        self.logger.info(f"Instance: {self.instance_number}")

    def simulation(self):
        for _ in range(self.number_simulation):
            self.logger = None
            self.logger = self.configure_logging()
            self.root = Node(
                self.initialise_root_node(),
                desired_selection_policy=self.desired_selection_policy,
                cp=self.cp,
            )
            self.best_leaf = None
            self.best_leaf_cost = float("inf")
            self.search()
            self.end_search_time = time.time() - self.start_time
            self.print_execution_times()
            self.get_final_nodes()
            self.print_characteristics_simulation()

    def select(self, node):
        self.logger.info("\nSELECTION\n")
        current_node = node
        self.logger.info(f"Starting selection at node: {current_node.state}")

        while current_node.children:
            self.logger.info(f"Current node: {current_node.state}")
            self.logger.info(f"Childrens: {current_node.children}")

            if not current_node.is_fully_expanded():
                # Select a random unvisited child if there are any
                unvisited_children = [
                    child for child in current_node.children if child.visit_count == 0
                ]
                self.logger.info(f"Unvisited children: {len(unvisited_children)}")
                if unvisited_children:
                    selected_child = random.choice(unvisited_children)
                    self.logger.info(
                        f"Randomly selected unvisited child: {selected_child}"
                    )
                    return True, selected_child

            else:
                current_node = current_node.best_child()
                self.logger.info(f"Moving to best child: {current_node.state}")
                # return True, current_node

        if (not current_node.children) and (
            current_node.state["current_day"] == self.number_of_areas
        ):
            self.logger.info("Final day selected")
            return False, current_node

        elif (not current_node.children) and (
            current_node.state["current_day"] != self.number_of_areas
        ):
            self.logger.info(f"The node {current_node.state} has no children")
            return False, current_node

        elif current_node.state["current_day"] == self.number_of_areas + 1:
            return True, current_node

    def expand_node(self, node):
        if node not in self.expanded_nodes:
            self.expanded_nodes.append(node)

            actions = self.possible_flights_from_an_airport_at_a_specific_day_with_previous_areas(
                node.state["current_day"],
                node.state["current_airport"],
                node.state["visited_zones"],
            )

            if node.state["current_day"] == self.number_of_areas:
                node.state["visited_zones"] = node.state["visited_zones"][1:]
                node.state["remaining_zones"].append(
                    self.associated_area_to_airport(self.starting_airport)
                )

                actions = self.possible_flights_from_an_airport_at_a_specific_day_with_previous_areas(
                    node.state["current_day"],
                    node.state["current_airport"],
                    node.state["visited_zones"],
                )

            expansion_policy = self.get_expansion_policy()
            actions = expansion_policy(actions)

            if actions:
                self.logger.info("Start expansion")
                for action in actions:
                    self.logger.info(f"{action}")
                    new_state = self.transition_function(node.state, action)
                    node.add_child(new_state)
                self.logger.info("End expansion")
            else:
                self.logger.info(f"No actions possible")
                return None

            return node

        else:
            self.logger.info("INFINITE LOOP")
            return None

    def search(self):

        duration = 10 * 60
        start_time = time.time()

        while time.time() - start_time < duration:
            while True:
                node_to_explore = self.select(self.root)

                self.logger.info(f"Node to explore: {node_to_explore[1].state}")

                if node_to_explore[1].state["current_day"] == self.number_of_areas + 1:
                    while not node_to_explore[1].parent.is_fully_expanded():
                        # self.logger.info(
                        #    "Node to explore is last day but all siblings have not been visited yet"
                        # )
                        node_to_explore = self.select(self.root)
                        self.logger.info(f"Node to explore: {node_to_explore[1].state}")
                        result = node_to_explore[1].state["total_cost"]
                        self.backpropagate(node_to_explore[1], result)

                    node_to_explore[1].state["visited_zones"].append(
                        self.associated_area_to_airport(
                            airport=node_to_explore[1].state["path"][-1]
                        )
                    )
                    return

                if not node_to_explore[0]:
                    expanded_node = self.expand_node(node=node_to_explore[1])
                    if not expanded_node:
                        self.logger.info("Not unexpandable so deleted")
                        node_to_explore[1].delete_node()
                        # self.logger.info(f"Nodes in tree: {len(self.collect_all_nodes())}")
                        if len(self.collect_all_nodes()) == 1:
                            self.logger.info(
                                "Everything has been deleted to the root node"
                            )
                            self.end_time_data_preprocessing = 0
                            self.end_search_time = 0
                            self.print_characteristics_simulation()
                            self.print_execution_times()
                            break
                        continue
                    else:
                        self.logger.info(
                            f"{node_to_explore[1].state} has been successfully expanded"
                        )
                        continue

                else:
                    simulation = self.simulate(node_to_explore[1])

                    if simulation[0]:
                        self.logger.info(f"Result from simulation: {simulation[0]}")

                        key = str(node_to_explore[1].state["current_day"])
                        value_to_add = simulation[0]
                        if key in self.simulations_dict:
                            self.simulations_dict[key].append(value_to_add)
                        else:
                            self.simulations_dict[key] = [value_to_add]

                        self.backpropagate(node_to_explore[1], simulation[0])

                    else:
                        self.logger.info(
                            "Simulation failed to reach a valuable state - node deleted"
                        )
                        self.delete_node(node_to_explore[1])
                        if len(self.collect_all_nodes()) == 1:
                            self.logger.info(
                                "Everything has been deleted to the root node"
                            )
                            self.end_time_data_preprocessing = 0
                            self.end_search_time = 0
                            self.print_characteristics_simulation()
                            self.print_execution_times()
                            break

    def simulate(self, node):
        self.logger.info("\n\nSIMULATION")
        simulation_policy = self.get_simulation_policy()
        current_simulation_state = deepcopy(node.state)
        self.logger.info(f"Selected node for simulation {current_simulation_state}")

        while current_simulation_state["current_day"] != self.number_of_areas:
            actions = self.possible_flights_from_an_airport_at_a_specific_day_with_previous_areas(
                day=current_simulation_state["current_day"],
                from_airport=current_simulation_state["current_airport"],
                visited_areas=current_simulation_state["visited_zones"],
            )

            action = simulation_policy(actions=actions)
            # self.logger.info(f"Action: {action}")
            if action is None:
                self.logger.info("Action is None")
                return False, False

            current_simulation_state = self.transition_function(
                current_simulation_state, action
            )
            # self.logger.info(f"Current simulation state {current_simulation_state}")

        if current_simulation_state["current_day"] == self.number_of_areas:
            current_simulation_state["visited_zones"] = current_simulation_state[
                "visited_zones"
            ][1:]
            current_simulation_state["remaining_zones"].append(
                self.associated_area_to_airport(self.starting_airport)
            )

            actions = self.possible_flights_from_an_airport_at_a_specific_day_with_previous_areas(
                day=current_simulation_state["current_day"],
                from_airport=current_simulation_state["current_airport"],
                visited_areas=current_simulation_state["visited_zones"],
            )

            if not actions:
                self.logger.info("No flight available to go back to the initial area")
                return False, False
            else:
                action = simulation_policy(actions=actions)

                current_simulation_state = self.transition_function(
                    current_simulation_state, action
                )
                self.logger.info(f"Current simulation state {current_simulation_state}")

                return current_simulation_state["total_cost"], current_simulation_state
