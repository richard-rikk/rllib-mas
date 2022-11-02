from turtle import width
import gym
import numpy as np

from .observation import MultiDiscreteMap
from typing import Tuple, Union, Dict, Any

class Simulation(gym.Env):
    """
    An Open AI Gym environment to train agents. During the
    simulation a 2D map is created where the agent, desctructuble
    and non-desctructuble objects (obstacles) and targets are randomly
    initialized.

    The state is a 1D flattened array of the map, while the action
    space is four actions. These are movement actions, but when 
    the agent would move to a desctructuble object position the 
    position will become `empty`. If the agent would move to a `target`
    position then a reward is given and the `target` is replaced 
    with `obstacle`.

    The simulation will run a maximum of `n`-steps.    
    """

    def __init__(self, config:Dict[str,Any]) -> None:
        """
        Initializes the class with randomly placed agent, targets
        obstacles, objects.

        # Arguments

        - height:int
        The height of the map.

        - width:int
        The width of the map.
        
        - config:Dict[str,Any]
        It is the configuration settings for the class. 
        It has to cointain the following keys:
            1. height:int - The map height.
            1. width:int - The map width.
            1. obj_cnt:int - The number of objects on the map.
            1. obst_cnt:int - The number of obstacles on the map.
            1. tar_cnt:int - The number of targets on the map.
            1. max_step:int - The number of allowed steps per game.

        # Returns
        - None

        # Errors
        - Throws value error if obj_cnt+obst_cnt+tar_cnt > height*width 
        """
        # Init the super class
        super().__init__()

        # Init our own variables
        cells                = config["height"] * config["width"]
        self.height          = config["height"]
        self.width           = config["width"]
        self.encode_empty    = 0
        self.encode_agent    = 1
        self.encode_object   = 2
        self.encode_obstacle = 3
        self.encode_target   = 4
        self.obj_cnt         = config["obj_cnt"]
        self.obs_cnt         = config["obst_cnt"]
        self.tar_cnt         = config["tar_cnt"]
        self.step_cnt        = 0
        self.max_step        = config["max_step"]

        # Init the variables of the super class
        self.observation_space = MultiDiscreteMap(
            height=self.height,
            width= self.width,
            max_value=self.encode_target,
            obj_cnt=self.obj_cnt, obs_cnt=self.obs_cnt,tar_cnt=self.tar_cnt,
            encoding=(self.encode_empty, self.encode_agent, 
            self.encode_object, self.encode_obstacle, self.encode_target),
          
        )
        self.action_space = gym.spaces.Discrete(4)
        self.action_cnt   = 4
    
    def __reset_needed(self) -> bool:
        """
        Determines if a reset is needed.
        A reset is needed if there are no more targets
        left or we exceeded the given max step count.
        # Arguments
        - None

        # Returns
        - If a reset is needed.

        # Errors
        - None
        """
        no_targets = self.observation_space.get_targets() == 0
        no_steps   = self.step_cnt >= self.max_step 
        return no_targets or no_steps
  
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment. All objects will
        be randomly placed on the map again.
      
        # Arguments
        - None

        # Returns
        - The starting state as a flattened numpy array.

        # Errors
        - None
        """
        self.step_cnt = 0
        self.observation_space.reset()

        return self.observation_space.sample()
    
    def step(self, action : Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, Dict[str,Any]]:
        """
        Transforms state `s` with action `a` resulting in `s'`. 
        If the agent moves to an `object` cell then clear the
        object. If the agent moves to an `empty` cell rewrite
        that cell. If the agent moves to a `target` cell
        rewrite it with `obstacle` cell.  
        # Arguments
        - action : np.ndarray | int
          The action to take in `s`. It can be 0,1,2,3. 
        # Returns
        - A tuple with the following:
          observation: np.ndarray
          reward: float
          done: bool
          info: Dict    
        # Errors
        - None
        """
        self.step_cnt += 1

        if action not in range(self.action_cnt):
            raise ValueError(f"The direction must be in {range(self.action_cnt)} but got {action}.")
        
        agent_coor, target_coor, in_bound = self.observation_space.step_target(dir=action)

        # What to do when the agent is out of bounds
        if not in_bound:
            return self.observation_space.sample(), 0., self.__reset_needed(), {}
        
        # Logic of the steps
        target_val = self.observation_space.get_value(c=target_coor)
        reward     = 0.
        if target_val == self.encode_empty:
            self.observation_space.set_value(c=agent_coor, value=self.encode_empty)
            self.observation_space.set_value(c=target_coor, value=self.encode_agent)
        elif target_val == self.encode_object:
            self.observation_space.set_value(c=target_coor, value=self.encode_empty)
        elif target_val == self.encode_obstacle:
            pass
        elif target_val == self.encode_target:
            self.observation_space.set_value(c=target_coor, value=self.encode_obstacle)
            reward = 1.
        
        return self.observation_space.sample(), reward, self.__reset_needed(), {}


    





