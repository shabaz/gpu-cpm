import gpucpm
import random
from skimage.segmentation import mark_boundaries
from PIL import Image
import numpy as np


def create_sorting_sim(dimension, nr_of_cells):
    temperature = 10
    
    sim = gpucpm.Cpm(dimension, 2, nr_of_cells, temperature, False)
    
    sim.set_constraints(cell_type = 1, lambda_area = 1, target_area = 40)
    sim.set_constraints(cell_type = 1, other_cell_type = 1, adhesion = 14)
    sim.set_constraints(cell_type = 0, other_cell_type = 1, adhesion = 16)
    
    sim.set_constraints(cell_type = 2, lambda_area = 1, target_area = 40)
    sim.set_constraints(cell_type = 2, other_cell_type = 2, adhesion = 2)
    sim.set_constraints(cell_type = 2, other_cell_type = 1, adhesion = 11)
    sim.set_constraints(cell_type = 2, other_cell_type = 0, adhesion = 16)

    return sim


def make_pic(types, ids, dimension):
    pic = np.ones((dimension,dimension,3)) * 255
    pic[types==1] = [200, 200, 200]
    pic[types==2] = [80,80,80]

    pic2 = mark_boundaries(pic, ids, color=(10,10,10), mode="inner")
    pic = (pic + pic2)/2
    pic = mark_boundaries(pic, types, color=(0,0,0), mode="inner")
    pic = mark_boundaries(pic, types!=0, color=(0,0,0), mode="outer")
    return pic.astype("uint8")



dimension = 256

newstate = np.zeros( (dimension,dimension), "int32" )

ncells = 1
for x in range(0,dimension-1,6):
    for y in range(0,dimension-1,6):
        if  (x-dimension/2)**2 + (y-dimension/2)**2 < (dimension**2)/9:
            t = random.randint( 1, 2 )
            newstate[x:x+3, y:y+3] = (t * (2**24)) + ncells
            ncells += 1

sim = create_sorting_sim(dimension, ncells)
state = sim.get_state()
state[:,:] = newstate

sim.push_to_gpu()

i = 1000

sim.run(cell_sync=0,block_sync=0,global_sync=1, 
            threads_per_block = 32,
            positions_per_thread = 4,
            positions_per_checkerboard = 2, 
            updates_per_checkerboard_switch = 1,
            updates_per_barrier = 1,
            iterations=i, 
            inner_iterations=i)

sim.synchronize()
sim.pull_from_gpu()

sim_state = sim.get_state()
types = sim_state // 2**24
ids = sim_state % 2**24

pic = make_pic(types, ids, dimension)
im = Image.fromarray(pic)

im.save("sorting_gpu_{}.png".format(i))


