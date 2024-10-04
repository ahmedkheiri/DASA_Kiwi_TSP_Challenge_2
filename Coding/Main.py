from MCTS import MCTS

expansion_policies = ["top_k", "ratio_k"]
simulation_policies = [
    "greedy_policy",
    "random_policy",
    "tolerance_policy",
]
selection_policy = ["UCB", "UCB1T"]
cp = [0, 1.41, 2 * 1.41]
ratio_expansion = [0, 0.3, 0.5, 0.7, 1]

instance_number = 1
root_dir = "Instances"
instance_path = f"./{root_dir}/{instance_number}.in"


mcts = MCTS(
    instance=instance_path,
    instance_number=instance_number,
    number_childrens=10,
    desired_expansion_policy=expansion_policies[1],
    ratio_expansion=ratio_expansion[0],
    desired_simulation_policy=simulation_policies[0],
    number_simulation=1,
    desired_selection_policy=selection_policy[0],
    cp=cp[1],
)

