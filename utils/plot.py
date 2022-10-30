import numpy as np
import matplotlib.pyplot as plt
from typing import List

def Plot(eps_len_mean: List[float], eps_reward_mean: List[float], eps_reward_max: List[float]) -> None:
    """
    Plot the training progresses.

    Arguments
    ---------
    eps_len_mean,eps_reward_mean, epis_reward_max:List[float]
    Stats for the agent during training. 

    Returns
    ---------
    None

    Errors
    ---------
    None

    """
    plt.figure(figsize=(20, 5))
    """
    
    plt.subplot(131)
    plt.title(f"Score: {np.mean(scores)}")
    plt.plot(scores)
    
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    
    plt.subplot(133)
    plt.title('epsilons')
    plt.plot(epsilons)
    
    plt.subplot(211)
    plt.title("Moves")
    plt.bar(range(len(moves)), list(moves.values()), tick_label=list(moves.keys()))
    """
    plt.subplot2grid((3,1), (0,0))
    plt.title(f"Episodes average score: {np.mean(eps_reward_mean)}")
    plt.plot(eps_reward_mean)

    plt.subplot2grid((3,1), (1,0))
    plt.title('Episodes average length')
    plt.plot(eps_len_mean)

    plt.subplot2grid((3,1), (2,0))
    plt.title('Episodes max score')
    plt.plot(eps_reward_max)

    plt.show()