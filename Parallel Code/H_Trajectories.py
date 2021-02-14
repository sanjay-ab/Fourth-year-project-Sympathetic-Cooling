#!/usr/bin/env python3
import concurrent.futures #for parallel processing
import time
from math import isnan
from sys import platform
from os import cpu_count
import itertools
from csv import writer 

import numpy as np
import scipy.constants as sc

from PWLibs.ARBInterp import tricubic  # Tricubic interpolator module
from PWLibs.Collisions import check_collisions  # Module for calculating collisions
from PWLibs.H_GS_E import Hzeeman  # Import the Hydrogen Zeeman energy calculator module
from PWLibs.Trap_Dist import makeH  # Module for making random atom distributions
from PWLibs.TrapVV import rVV  # Velocity Verlet numerical integrator module
from PWLibs.Plotting import plotting #imports plotting module.

mH = 1.008*sc.physical_constants["atomic mass constant"][0] # mass H
Nli = 1e7 # Number of lithium atoms. (max value is 7.5e9 due to max density of 1e17m^-3)
gamma = 1e-3 #inelastic to elastic collision ratio, default is 1e-3
dt = 1e-6	# timestep of simulation
t_end = 0.1	# when do we want to stop the simulation
times_to_save=[] #list times in the simulation in which to save the data
number_of_saves = len(times_to_save)
timer = time.perf_counter()


# Location of the field source file, depending on operating platform
if platform == "win32":
	slash = "\\"
elif platform == "linux" or "linux2":
	slash = "/"

trapfield=np.genfromtxt(f"Input{slash}SmCo28.csv", delimiter=',') # Load MT-MOT magnetic field
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
	pointer = 0

	while t < t_end:

		atom_array = rVV(HU11Interp, atom_array, dt, mH)	# this uses the imported velocity verlet integrator to iterate the particle positions and velocities

		atom_array = check_collisions(atom_array, Nli, dt, gamma) #checks whether a collision occurs and carries out the calculation if one does.

		t += dt	# increment time elapsed

		if time.perf_counter()- loop > 60:	# every 60 seconds this will meet this criterion and run the status update code below
			print(f'Chunk {index} loop ' + '{:.1f}'.format(100 * (t / t_end)) + ' % complete. Time elapsed: {}s'.format(int(time.perf_counter()-timer)))	# percentage complete
			loop = time.perf_counter()	# reset status counter

		if pointer != number_of_saves:
			if t>times_to_save[pointer]: #save data to file at specified time through iteration
				with open(f"Output{slash}H_end gamma={gamma} t={times_to_save[pointer]} dt={dt} Nli=" + format(Nli,".1e") + ".csv",'a+',newline='') as outfile:
					csv_writer = writer(outfile)
					for row in atom_array:
						csv_writer.writerow(row)

				pointer += 1

	print("\n Chunk {} of {} complete. \n Time elapsed: {}s.".format(index, n_chunks, int(time.perf_counter()-timer))) #print index to get a sense of how far through the iteration we are
	return atom_array #return the chunk

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
		to run in parallel with each other. It yields the return value of the function iterate which can be extracted from "results" by iterating through. Above, we create the chunks 
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
	print(f"Time elapsed: {int(time.perf_counter()-timer)}s")  #can't just use time.perf_counter() since this is not equal to the simulation time when executing on the supercomputer

	print("Saving output")
	np.savetxt(f"Output{slash}H_init.csv", HRange_init, delimiter=',')
	np.savetxt(f"Output{slash}H_end gamma={gamma} t={t_end} dt={dt} Nli=" + format(Nli,".1e") + ".csv", HRange, delimiter=',')
	print("Done")

	print("Graphing output")
	plotting(HRange, HRange_init) 
	print("Done")

if __name__ == "__main__":
	main()
