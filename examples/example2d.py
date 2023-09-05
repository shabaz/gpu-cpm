import gpucpm
from PIL import Image
import numpy as np


dimension = 256
sim = gpucpm.Cpm(dimension, 2, 10, 20, False)

sim.add_cell(1, 100,100)
sim.add_cell(1, 200,200)
sim.add_cell(1, 200,100)
sim.set_constraints(cell_type=1, lambda_area = 10, target_area = 200)
sim.set_constraints(cell_type = 0, other_cell_type = 1, adhesion = 10)

sim.push_to_gpu()
sim.run(cell_sync=0,block_sync=0,global_sync=1, 
    threads_per_block = 4,
    positions_per_thread = 8,
    positions_per_checkerboard = 4, 
    updates_per_checkerboard_switch = 1,
    updates_per_barrier = 1,
    iterations=2000, 
    inner_iterations=2000)

#make sure gpu kernel finished
sim.synchronize()

# this call only retreives cell center positions from GPU, especially efficient
# when running a large simulation, because memory transfer of whole sim state
# to system memory is avoided
centroids = sim.get_centroids()
print(centroids)

# retrieve full simulation state, to make image of simulation snapshot
sim.pull_from_gpu()
state = sim.get_state()
cell_types = state // 2**24
cell_ids = state % 2**24

# make image
pic = np.ones((dimension,dimension,3)) * 255
pic[cell_ids!=0] = 0
pic = pic.astype("uint8")
im = Image.fromarray(pic)
im.save("2d_simulation.png")
