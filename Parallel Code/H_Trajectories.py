import concurrent.futures #for parallel processing
import time
from math import isnan
from sys import platform
from os import cpu_count
import itertools 

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from mpl_toolkits.mplot3d import Axes3D  # allows graphing in 3D

from PWLibs.ARBInterp import tricubic  # Tricubic interpolator module
from PWLibs.collisions import checkProbLi  # Module for calculating collisions
from PWLibs.H_GS_E import Hzeeman  # Import the Hydrogen Zeeman energy calculator module
from PWLibs.Trap_Dist import makeH  # Module for making random atom distributions
from PWLibs.TrapVV import rVV  # Velocity Verlet numerical integrator module

mH = 1.008*sc.physical_constants["atomic mass constant"][0] # mass H
Nli = 1e7 # Number of lithium atoms.
dt = 1e-6	# timestep of simulation
t_end = 1	# when do we want to stop the simulation

# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"
	
trapfield=np.genfromtxt(f"Input{slash}Trap.csv", delimiter=',') # Load MT-MOT magnetic field
trapfield[:,:3]*=1e-3 # Modelled the field in mm for ease, make ;it m
# This is a vector field but the next piece of code automatically takes the magnitude of the field

HEnergies = Hzeeman(trapfield)	# this instantiates the H Zeeman energy class with the field
# This class instance now contains 4 Zeeman energy fields for the different substates of H
# For now we are only interested in the F=1, mF=1 state, HEnergies.U11
# Others are U00, U1_1, U10

HU11Interp = tricubic(HEnergies.U11,'quiet') #This creates an instance of the tricubic interpolator for this energy field
# We can now query arbitrary coordinates within the trap volume and get the interpolated Zeeman energy and its gradient

def iterate(atom_array, index, n_chunks):
	"""Function where computation for chunks takes place"""

	t = 0 #initialise time variable counter
	loop = time.perf_counter()

	while t < t_end:

		atom_array = rVV(HU11Interp, atom_array, dt, mH)	# this uses the imported velocity verlet integrator to iterate the particle positions and velocities

		atom_array = checkProbLi(atom_array, Nli, dt) #checks whether a collision occurs and carries out the calculation if one does.

		t += dt	# increment time elapsed

		if time.perf_counter()- loop > 60:	# every 60 seconds this will meet this criterion and run the status update code below
			print(f'Chunk {index} loop ' + '{:.1f}'.format(100 * (t / t_end)) + ' % complete. Time elapsed: {}s'.format(int(time.perf_counter())))	# percentage complete
			loop = time.perf_counter()	# reset status counter

	print("\n Chunk {} of {} complete. \n Time elapsed: {}s.".format(index, n_chunks, int(time.perf_counter()))) #print index to get a sense of how far through the iteration we are
	return atom_array #return the chunk

def plotting(HRange, HRange_init):
	"""Function takes care of graphing the result of the simulation"""

	init_speeds = [np.linalg.norm(HRange_init[i,3:]) for i in range(len(HRange_init))] #initial speeds of hydrogen atoms
	speeds = [np.linalg.norm(HRange[i,3:]) for i in range(len(HRange))] #final speeds of hydrogen atoms

	kinetic_energy = [0.5* mH* s**2 for s in speeds] #final kinetic energy of hydrogen atoms
	init_kinetic_energy = [0.5* mH* s**2 for s in init_speeds] #initial kinetic energy of hydrogen atoms

	colour = colours(speeds,max(speeds)) #colour array to demonstrate final speeds of particles graphically
	init_colour = colours(init_speeds,max(init_speeds)) #colour array used to demonstrate initial speeds of particles graphically

	print(f"Number of hydrogen left in trap: {len(HRange)}")

	plot2 = plt.figure('3D Final Positions of Particles')
	ax2 = plot2.add_subplot(111, projection='3d') #plot final positions of particles in 3d while colour coding speeds 
	ax2.set_xlabel("x(mm)", fontsize = 10)
	ax2.set_ylabel("y(mm)", fontsize = 10)
	ax2.set_zlabel("z(mm)", fontsize = 10)
	ax2.set_xlim3d([-10,10])
	ax2.set_ylim3d([-10,10])
	ax2.set_zlim3d([-10,10])
	ax2.scatter(HRange[:,0]*1000,HRange[:,1]*1000,HRange[:,2]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Final Speeds of hydrogen particles')
	plt.xlabel("Speed ($ms^{-1}$)", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(speeds, bins= "auto", zorder = 1,density = True) #histogram of final speeds of hydrogen atoms.

	_ = plt.figure('Initial Speeds of hydrogen particles')
	plt.xlabel("Speed ($ms^{-1}$)", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(init_speeds, bins= "auto", zorder = 1,density = True) #histogram of initial speeds of hydrogen atoms.

	_ = plt.figure('Final Kinetic Energy of hydrogen particles')
	plt.xlabel("$log_{10}$(Kinetic Energy (J))", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(np.log10(kinetic_energy), bins= "auto", zorder = 1,density = True) #histogram of final kinetic energy of hydrogen atoms.

	_ = plt.figure('Initial Kinetic Energy of hydrogen particles')
	plt.xlabel("$log_{10}$(Kinetic Energy (J))", fontsize = 10)
	plt.ylabel("Normalized Counts", fontsize = 10)
	plt.hist(np.log10(init_kinetic_energy), bins= "auto", zorder = 1,density = True) #histogram of initial kinetic energy of hydrogen atoms.

	_ = plt.figure('Final Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.xlim([-10,10])
	plt.ylim([-10,10])
	plt.scatter(HRange[:,0]*1000,HRange[:,1]*1000, c=colour, s=0.5) #plots positions of final particles and colour codes them according to their speeds.

	_ = plt.figure('Initial Positions of Particles')
	plt.xlabel("x(mm)", fontsize = 10)
	plt.ylabel("y(mm)", fontsize = 10)
	plt.xlim([-10,10])
	plt.ylim([-10,10])
	plt.scatter(HRange_init[:,0]*1000,HRange_init[:,1]*1000, c=init_colour, s=0.5) #plots positions of initial particles and colour codes them according to their speeds.

	_ = plt.figure('Final Phase - Space Distribution of Particles')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(HRange[:,2]*1000,HRange[:,5],s=1) #plots final phase space distn of atoms

	_ = plt.figure('Initial Phase - Space Distribution of Particles')
	plt.xlabel("z(mm)", fontsize = 10)
	plt.ylabel("Velocity ($ms^{-1}$)", fontsize = 10)
	plt.scatter(HRange_init[:,2]*1000,HRange_init[:,5],s=1) #plots initial phase space distn of atoms.

	plt.show()

def colours(speeds,maxSpeed):
	"""function that creates a simple colour key based on speeds for showing speed in plotting"""
	colour = []

	for  item in speeds: #create an array for colour coding graphs.
		colour.append([float(item/maxSpeed), 0 ,float(1-item/maxSpeed)]) #colour moves from blue (cold) to red (hot)

	return colour 

def main():

	HRange_init = np.genfromtxt(f"Input{slash}H_init t=1e-4 dt=1e-6.csv", delimiter=',') #take initial hydrogen distn 

	print("Solving particle motion:")
	#moves all particles velocities and postions through a time dt. Uses the velocity verlet integration since we need to integrate over the potential to find the path of the particles.
	#we also need the interpolator since the field is not known continuously we need to need to interpolate between known points to get the whole field

	#use concurrent.futures module to calculate the paths of each hydrogen particle in parallel for faster execution
	chunks_per_workers = 1 #number of chunks per worker - vary between 1 and 40 for efficient execution depending on the length of the simulation
	n_chunks = chunks_per_workers * cpu_count() #number of chunks (cpu_count yields the number of logical processors on the system)
	indicies = np.arange(1,n_chunks+1) #used to show which chunk we are currently on
	chunks = [HRange_init[slice(int((len(HRange_init)/n_chunks)*i),int((len(HRange_init)/n_chunks)*(i+1)))] for i in range(n_chunks)] #split array into chunks to be computed in parallel.

	with concurrent.futures.ProcessPoolExecutor() as executor: #use with statement so that executor automatically closes after completion
		"""executor.map maps the iterables in the second parameter to the function "iterate" similar to the inbuilt map function although this allows multiple versions of the function
		to run in parallel with each other. It yields the return value of the function iterate which can be extracted from "results" by iterating thorugh. Above, we create the chunks 
		we want to execute with rather than using the chunksize parameter. This reduces the overhead in starting and stopping parallel processes by controlling the number of processes created."""

		results = executor.map(iterate, *(chunks,indicies,itertools.repeat(n_chunks,n_chunks))) #result of simulation
	
	HRange = [] #initialise list to hold final distributions of atoms

	for result in results:
		#loop through the array of arrays of atoms
		for res in result:
			#loop through each atom in each array of atoms
			HRange.append(res) #append each atom to list

	HRange = np.asarray(HRange) #turn list into 2d numpy array

	print("Done")
	print(f"Time elapsed: {int(time.perf_counter())}s")  #time.perf_counter() is approximately equal to the execution time.

	print("Saving output")
	np.savetxt(f"Output{slash}H_init.csv", HRange_init, delimiter=',')
	np.savetxt(f"Output{slash}H_end.csv", HRange, delimiter=',')
	print("Done")

	print("Graphing output")
	plotting(HRange, HRange_init) 
	print("Done")

if __name__ == "__main__":
	main()
