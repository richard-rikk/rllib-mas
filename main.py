from simulation import Simulation
from model      import Training

# sim2 = Simulation(config={"height":5, "width":5, "obj_cnt":0, "obst_cnt":0, "tar_cnt":1, "max_step":10})
# for i in range(10):
#     print(sim2.step(action=3))
#     print(sim2.observation_space.map)
#     print("\n")

Training(
    epochs=1_000,
    num_workers=4,
    buffer_size=1024,
    gpus=1,
    sim_height=10,
    sim_width=10,
    sim_objs=10,
    sim_obs=10,
    sim_tars=10,
    max_step=100,
)



