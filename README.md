# Fourth-year-project
Code that I've written for my 4th year project on simulating the sympathetic cooling of atomic hydrogen with ultra-cold lithium atoms

Only two of the files for the program have been uploaded. The additional libraries were written by a PhD student, not by me and so have not been included. Similarly the additional data required for the program to run has also not been included.

The program works by simulating the paths of hydogen particles in a magnetic trap by changing their positions and velocities every timestep, dt. Then it checks for the probability of a collision with a cloud of lithium atoms, if a collision occurs then a random velocity is chosen for the lithium atom and an elastic collision is computed. 

There are two sets of code, one for execution in parallel and one for sequential execution. The parallel code run significantly faster than the sequential code, although it does lack some functionality like being able to track the temperature and particle number of the lithium atom cloud, although, this is fairly insignificant anyway.
