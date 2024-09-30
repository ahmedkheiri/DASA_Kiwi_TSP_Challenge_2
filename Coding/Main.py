from MCTS import MCTS
from Node import Node
from Data_Preprocessing import data_preprocessing


expansion_policies = ["top_k", "ratio_k"]
simulation_policies = [
    "greedy_policy",
    "random_policy",
    "tolerance_policy",
]
selection_policy = ["UCB", "UCB1T"]


instance_number = 4
root_dir = "Flight connections dataset" - save instances in this folder
instances = f"insert_your_path_to_root_dir/{root_dir}"
instance_path = f"{instances}/{instance_number}.in"


mcts = MCTS(
    instance=instance_path,
    instance_number=instance_number,
    number_childrens=10,
    desired_expansion_policy="ratio_k",
    ratio_expansion=0,
    desired_simulation_policy="random_policy",
    number_simulation=1,
    desired_selection_policy="UCB",
    cp=0,
)
