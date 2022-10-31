import gym
import numpy as np

from gym                    import logger
from gym.spaces.space       import Space
from gym.spaces.discrete    import Discrete
from typing                 import List, Tuple, Optional, Dict, Any, Sequence,Iterable

class MultiDiscreteMap(Space):
    """
    Creates a 2D map with multiple discrete values
    on it. Although, this class uses a 2D numpy array
    as a map, during sampling it will yield arrays
    that correspond to the return value of the 
    `__preprocess` function. Therefore, these will
    be flattened, one-hot encoded arrays.
    """
    def __init__(self, height:int, width:int, max_value:int,
        obj_cnt:int, obs_cnt:int, tar_cnt:int,
        encoding:Tuple[int,int,int,int,int],
    ) -> None:
        """
        Initializes the class with randomly placed agent, targets
        bstacles, objects.

        # Arguments
        - height: int
        The height of the map.
        - width: int
        The width of the map.
        - max_value: int
        The maximum discrete value on the map.
        - obj_cnt: int
        The number of objects on the map.
        - obs_cnt: int
        The number of obstacles on the map.
        - tar_cnt: int
        The number of targets on the map.
        - encoding:Tuple[int,int,int,int,int]
        Contains the discrete values to use. In order these
        have to be: `empty`, `agent`, `object`, `obstacle`,
        and `target`.      

        # Returns
        - None

        # Errors
        - Throws value error if obj_cnt+obst_cnt+tar_cnt > height*width 
        """
        # Check exceptions (+1 is the agent)
        if (all_obj := obj_cnt + obs_cnt + tar_cnt + 1) > height*width:
            raise ValueError(f"Cannot place {all_obj} objects on {height*width} cells.")
        
        self.height          = height
        self.width           = width
        self.cells           = height * width
        self.max_value       = max_value
        self.min_value       = 0
        self.objects         = obj_cnt
        self.obstacles       = obs_cnt
        self.targets         = tar_cnt
        self.encode_empty    = encoding[0]
        self.encode_agent    = encoding[1]
        self.encode_object   = encoding[2]
        self.encode_obstacle = encoding[3]
        self.encode_target   = encoding[4]
        nvec                 = self.__preprocess(np.array([max_value] * self.cells))

        super().__init__(nvec.shape, dtype=np.uint8)

        self.map = self.__init_map()

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
        map[indxs[:self.objects]] = self.encode_object
        indxs = indxs[self.objects:]

        map[indxs[:self.obstacles]] = self.encode_obstacle
        indxs = indxs[self.obstacles:]

        map[indxs[:self.targets]] = self.encode_target
        indxs = indxs[self.targets:]

        pos      = indxs[0]
        map[pos] = self.encode_agent

        # Return a 2D map
        return map.reshape((self.height, self.width))
    
    def step_target(self,dir:int) -> Tuple[Tuple[int,int], bool]:
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
        - None
        """
        
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
    
    def reset(self) -> None:
        """
        Resets the map with the given object counts.
        It will generate a new random map, after calling
        this function the prevous state is gone.
        # Arguments
        - None

        # Returns
        - None

        # Errors
        - None
        """
        self.map = self.__init_map()
    
    def get_value(self, c:Tuple[int,int]) -> int:
        """
        Returns the value at the given coordinate.
        
        # Arguments
        - c: Tuple[int,int]
        The target coordinate.

        # Returns
        - value: int
        The value to rewrite current value.

        # Errors
        - None
        """

        return self.map[c]
    
    
    def set_value(self, c: Tuple[int,int], value:int) -> None:
        """
        Changes a value on the map given a coordinate.

        # Arguments
        - c: Tuple[int,int]
        The target coordinate to change.
        - value: int
        The value to rewrite current value.

        # Returns
        - None

        # Errors
        - None
        """
        self.map[c] = value
    
    def get_targets(self) -> int:
        """
        Returns the number of active targets left.
        """
        return len(self.map[self.map == self.encode_target])
    
    def __preprocess(self, matrix:np.ndarray) -> np.ndarray:
        """
        Does preprocessing on the map:
        1. One-hot encodes the map
        1. Flattenes the map.
        """
        ncols = self.max_value + 1  # The length of the onehot encoded tensor.
        out = np.zeros( (matrix.size , ncols), dtype=np.uint8) # Create the zero matrix.
        out[ np.arange(matrix.size) , matrix.ravel() ] = 1  # Change one value to a one based on the coordinates.
        #out.shape = self.map.shape + (ncols,) # Reshape.

        return out.flatten()

    
    def sample(self, mask: Optional[Tuple] = None) -> np.ndarray:
        """
        Returns the state of the map as a 1D array. Use this
        function if the state needs some preprocessing. Use
        this function after calling reset as well.
        """
        return self.__preprocess(matrix=self.map)
    
    # The super class functions
    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than :class:`gym.Space` - never None."""
        return self._shape  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True
    
    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if isinstance(x, Sequence):
            x = np.array(x)  # Promote list to array for contains check

        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return bool(
            isinstance(x, np.ndarray)
            and x.shape == self.shape
            and x.dtype != object
            and np.all(self.min_value <= x)
            and np.all(x < self.max_value)
        )
    
    def to_jsonable(self, sample_n: Iterable[np.ndarray]) -> List[List]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return [sample.tolist() for sample in sample_n]

    def from_jsonable(self, sample_n) -> np.ndarray:
        """Convert a JSONable data type to a batch of samples from this space."""
        return np.array(sample_n)

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"MultiDiscreteMap:\n{self.map}"
      
    def __len__(self) -> int:
        """Gives the ``len`` of samples from this space."""
        if self.map.ndim >= 2:
            logger.warn(
                "Getting the length of a multi-dimensional MultiDiscrete space."
            )
        return len(self.map)

    def __eq__(self, other) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return isinstance(other, MultiDiscreteMap) and np.all(self.map == other.map)
    
    def __getitem__(self, index) -> Discrete:
        """
        Extract a subspace from this ``MultiDiscrete`` space.
        For us it doesn't matter since all of the indexes can
        take up exactly the same discrete value.
        """        
        return Discrete(self.max_value)

        
        