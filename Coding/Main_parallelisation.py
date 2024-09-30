from MCTS_test_parralelisation import MCTS
from Node import Node
from Data_Preprocessing import data_preprocessing
from Logs_process import logs_analysis


def run_mcts():
    expansion_policies = ["top_k", "ratio_k"]
    simulation_policies = [
        "greedy_policy",
        "random_policy",
        "tolerance_policy",
    ]
    selection_policy = ["UCB", "UCB1T", "SP", "Bayesian"]

    instance_number = 4
    root_dir = "Flight connections dataset"
    instances = f"/Users/adslv/Documents/LU/Term 3/Kiwi_TSP_Challenge/Code/{root_dir}"
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

    # logs_analysis(root_dir=instances, create_or_not=True, do_we_delete=False)


if __name__ == "__main__":
    run_mcts()
