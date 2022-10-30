import gym
import numpy as np

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
    
        # Check exceptions (+1 is the agent)
        if (all_obj := self.obj_cnt + self.obs_cnt + self.tar_cnt + 1) > self.height*self.width:
            raise ValueError(f"Cannot place {all_obj} objects on {cells} many cells.")

        # Init the variables of the super class
        self.observation_space= gym.spaces.Box(
            np.zeros(shape=cells, dtype=np.uint8),
            np.ones(shape=cells, dtype=np.uint8) * 4,
            dtype=np.uint8,
        )

        self.action_space = gym.spaces.Discrete(4)
        self.action_cnt   = 4
        self.map          = self.__init_map()
    
    def __init_map(self) -> np.ndarray:
        """
        Initializes the 2D map of the state with the
        proper amount of objects on it.

        # Arguments
        - None

        # Returns
        - A 2D array representation of the map.

        # Errors
        - None
        """
        # Create an empty 1D map.
        cells = self.height * self.width
        map   = np.zeros(shape=cells, dtype=np.uint8)

        # Create the indicies and shuffle them
        
        indxs = np.arange(start=0, stop=cells, step=1, dtype=np.uint32)
        np.random.shuffle(indxs)

        # Place objects, obstacles, targets and agent in this order.
        map[indxs[:self.obj_cnt]] = self.encode_object
        indxs = indxs[self.obj_cnt:]

        map[indxs[:self.obs_cnt]] = self.encode_obstacle
        indxs = indxs[self.obs_cnt:]

        map[indxs[:self.tar_cnt]] = self.encode_target
        indxs = indxs[self.tar_cnt:]

        map[indxs[0]] = self.encode_agent

        # Return a 2D map
        return map.reshape((self.height, self.width))
    
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
        no_targets = len(self.map[self.map == self.encode_target]) == 0
        no_steps   = self.step_cnt >= self.max_step 
        return no_targets or no_steps
    
    def __return_obs(self) -> np.ndarray:
        """
        Returns the observation of the map. For now,
        the map will be only flattened.
        
        # Arguments
        - None

        # Returns
        - A flattened version of the map.

        # Errors
        - None
        """

        return self.map.flatten()
    
    def __step_target(self,dir:int) -> Tuple[Tuple[int,int], bool]:
        """
        Finds the target of the step based on the direction. Also,
        does boundry check on the target coordinate.

        # Arguments
        - dir:int
        The direction to move in, it must be 0,1,2,3. 0 is up,
        1 is right, 2 is down, 3 is left.

        # Returns
        - The coordinate of the agent and target cells and returns
        true if the target coordinate is valid.

        # Errors
        - Throws error if dir is not in 0,1,2,3.
        """

        if dir not in range(self.action_cnt):
            raise ValueError(f"The direction must be in {range(self.action_cnt)} but got {dir}.")
        
        agent_pos = np.where(self.map == self.encode_agent)
        x         = agent_pos[0][0]
        y         = agent_pos[1][0]
        x_        = x
        y_        = y

        if dir == 0:
            x -= 1
        elif dir == 1:
            y += 1
        elif dir == 2:
            x += 1
        else:
            y -= 1
        
        in_bound = x >= 0 and y >= 0 and x < self.height and y < self.width

        return ((x_,y_), (x,y), in_bound)
    
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
        self.map      = self.__init_map()

        return self.__return_obs()
    
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
        agent_coor, target_coor, in_bound = self.__step_target(dir=action)

        # What to do when the agent is out of bounds
        if not in_bound:
            return self.__return_obs(), 0., self.__reset_needed(), {}
        
        # Logic of the steps
        target_val = self.map[target_coor]
        reward     = 0.
        if target_val == self.encode_empty:
            self.map[agent_coor]  = self.encode_empty
            self.map[target_coor] = self.encode_agent
        elif target_val == self.encode_object:
            self.map[target_coor] = self.encode_empty
        elif target_val == self.encode_obstacle:
            pass
        elif target_val == self.encode_target:
            self.map[target_coor] = self.encode_obstacle
            reward = 1.
        
        return self.__return_obs(), reward, self.__reset_needed(), {}


    





