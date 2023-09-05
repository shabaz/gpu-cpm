import gpucpm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


dimension = 64
sim = gpucpm.Cpm(dimension, 3, 10, 20, False)

sim.add_cell(1, 16, 16, 16,)
sim.add_cell(1, 48, 48, 32,)
sim.add_cell(1, 32, 16, 48,)
#sim.add_cell(1, 200,200, 100)
#sim.add_cell(1, 200,100, 200)
sim.set_constraints(cell_type=1, lambda_area = 10, target_area = 2000)
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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(cell_ids != 0, edgecolor="k")
plt.savefig('3d_simulation.png',bbox_inches='tight')
