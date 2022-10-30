import os
import ray
import torch
from ray.rllib.algorithms import apex_dqn
from simulation           import Simulation
from utils                import Plot

def Training(
    epochs:int, num_workers:int, gpus:int,
    sim_height:int, sim_width:int, sim_objs:int, sim_obs:int, sim_tars:int  
    ) -> None:
    """
    Trains the agent with the environment.

    # Arguments
    - epochs:int 
    The number of games to play in the enviroment.

    - num_workers:int
    The number of agents to use for training.

    -gpus:int
    The number of GPUs to use.

    -sim_height:int, sim_width:int, sim_objs:int, sim_obs:int, sim_tars:int 
    The settings for the simulation. In order it is the
    height and width of the map, the number of objects,
    the number of obstacles, the number of targets.
    """
    # Init ray
    ray.init()

    # Set the config.
    config = apex_dqn.ApexDQNConfig()
    config.training(
        num_atoms=2, v_min=0., v_max=10.,
        noisy=True, dueling=True, hiddens=[128,128],
        double_q=True, n_step=5
    )
    config.resources(num_gpus=gpus)
    config.framework(framework="torch")
    config.environment(
        env=Simulation, 
        env_config={
            "height":sim_height,
            "width":sim_width,
            "obj_cnt":sim_objs,
            "obst_cnt":sim_obs,
            "tar_cnt":sim_tars,
            }
        )
    config.num_workers = num_workers

    # Set the model
    model = apex_dqn.ApexDQN(env=Simulation, config=config)

    # Set the training loop
    epoch                 = epochs
    max_mean_score        = 0.
    episode_len_mean      = []
    episode_reward_mean   = []
    episode_reward_max    = []
    save_dir              = "save/"

    for i in range(epoch):
        result = model.train()

        # Save metrics
        episode_len_mean.append(result["episode_len_mean"])
        episode_reward_mean.append(result["episode_reward_mean"])
        episode_reward_max.append(result["episode_reward_max"])

        if episode_reward_mean[-1] > max_mean_score:
            max_mean_score = episode_reward_mean[-1]

        print(f"Epoch finished: {i+1} -- Mean reward: {episode_reward_mean[-1]} -- Max mean: {max_mean_score}")

    model.save(checkpoint_dir=save_dir)
    Plot(eps_len_mean=episode_len_mean, eps_reward_mean=episode_reward_mean, eps_reward_max=episode_reward_max)

    # Shutdown the ray instance.
    ray.shutdown()

def Save(model:torch.nn.Module, name:str,loc:str) -> None:
    """
    Saves the model to a location.
    Arguments
    ---------
    name:str
        The name of the model.
    loc:str
        The model will be saved here.
    Returns
    ---------
    None
    Errors
    ---------
    None
    """
    # Have to use this class TorchPolicy(Policy):
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(os.path.join(loc, name+".pt"))# Save

    
def Load(name:str,loc:str) -> torch.nn.Module:
    """
    Loads a model from file.
    
    Arguments
    ---------
    loc:str
        The model will be loaded from here.
    
    Returns
    ---------
    The loaded model.
    
    Errors
    ---------
    None
    """
    model = torch.jit.load(os.path.join(loc, name+".pt"))
    model.eval()
    
    return model


