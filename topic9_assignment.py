import numpy as np
from scipy.optimize import linprog

def calculate_nash_equilibrium(payoff_matrix):
    """
    Return known Nash Equilibrium for classic games (educational only).
    """
    prisoners_dilemma = np.array([[(-1, -1), (-3, 0)], [(0, -3), (-2, -2)]])
    if np.array_equal(payoff_matrix, prisoners_dilemma):
        return ("Defect", "Defect")
    else:
        return None

def solve_zero_sum_game(payoff_matrix):
    """
    Solve a zero-sum game using linear programming.
    """
    m, n = payoff_matrix.shape
    payoff_matrix = payoff_matrix - np.min(payoff_matrix) + 1  # make all values positive

    c = [-1] * m
    A_ub = -payoff_matrix.T
    b_ub = [-1] * n
    A_eq = [[1] * m]
    b_eq = [1]
    bounds = [(0, None) for _ in range(m)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    if res.success:
        game_value = -res.fun + np.min(payoff_matrix) - 1  # revert the shift
        return game_value, res.x
    else:
        print("Failed to solve the game.")
        return None

def simulate_prisoners_dilemma(strategies, iterations):
    """
    Simulate Iterated Prisoner's Dilemma.
    """
    outcomes = []
    history1, history2 = [], []
    for _ in range(iterations):
        move1 = strategies[0](history1, history2)
        move2 = strategies[1](history2, history1)
        history1.append(move1)
        history2.append(move2)
        outcomes.append((move1, move2))
    return outcomes

# Sample strategy functions
def always_cooperate(my_history, opponent_history):
    return "C"

def always_defect(my_history, opponent_history):
    return "D"

# Example usage:
if __name__ == "__main__":
    print("Nash Equilibrium Example:")
    print(calculate_nash_equilibrium(np.array([[(-1, -1), (-3, 0)], [(0, -3), (-2, -2)]])))

    print("\nZero-Sum Game Example:")
    payoff = np.array([[1, -1], [-1, 1]])
    print(solve_zero_sum_game(payoff))

    print("\nPrisoner's Dilemma Simulation:")
    results = simulate_prisoners_dilemma((always_cooperate, always_defect), 5)
    print(results)
