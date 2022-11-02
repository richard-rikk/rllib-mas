from simulation import Simulation
from model      import Training

# sim2 = Simulation(config={"height":10, "width":10, "obj_cnt":10, "obst_cnt":10, "tar_cnt":10, "max_step":10})
# for i in range(10):
#     print(sim2.step(action=3))
#     print(sim2.observation_space.map)
#     print("\n")

Training(
    epochs=2,
    num_workers=4,
    buffer_size=2048,
    gpus=0,
    sim_height=10,
    sim_width=10,
    sim_objs=10,
    sim_obs=10,
    sim_tars=10,
    max_step=100,
)



