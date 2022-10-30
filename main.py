from simulation import Simulation
from model      import Training
from utils      import Plot
# sim2 = Simulation(**{"height":5, "width":5, "obj_cnt":0, "obst_cnt":0, "tar_cnt":0})

Training(
    epochs=2,
    num_workers=4,
    gpus=0,
    sim_height=10,
    sim_width=10,
    sim_objs=10,
    sim_obs=10,
    sim_tars=10
)



