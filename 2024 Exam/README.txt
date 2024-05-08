2024 Exam B151138

run deterministic_exam.py for parts 1a-1c.
run sequential_exam.py for parts 1d-1f.

1b. For n=2, the simulation presents initial gliding structures that move along the horizontal and vertical axes of the grid. These gliders subsequently evolve the grid into a state of dynamic equlibrium (steady state), where the grid is populated with oscillating (blinker) structures. For n=3, The grid initialization leads to a sink state, as the 'on' probability is too small for a significant fraction of the initially off states to have exactly 3 'on' neighbours. From the plot of on states vs. time for n=2, The number of on states increases to a steady state value of around 2000 on states after a time of approximately 100 iterations. From the attached .gif file (q1b_visualization_n2.gif), the early and the steady state behaviour can be seen for n=2.

1c. For n=2, all on states inside the cube instantly die. At the edges of the cube, no on states are created as here the number of initially on neighbours exceeds 2 for off states touching the cube edges. The only surviving states are those touching the corners of the cube (as here number of neighbours can equal 2 for off cells positioned outside the cube). These states then transform into stationary oscillators, which remain and form the dynamic equilibrium for the system. For n=3, the result is a sink state, where the initial behaviours forms a crosshair/target pattern of on states, before these die to form a completely dead grid after approx. 10 iterations.

1e. The two phases that can be seen are separated along the diagonal of the heatmap. One such phase corresponds to a sink phase where the value of p1 is (quite) larger than the value of p2. In this case, on states are more likely to turn off than off states are likely to turn on (given the neighbour conditions). This results in death of the grid (so that it is entirely off states). The other phase is where p2 is greater than p1. In this case, dead cells are more likely to come alive again (given the neighbour conditions) compared to alive cells turning off. In this case, the grid eventually reaches a point of dynamic equilibrium, where the total number of on states is greater than zero, but stable. 

1f. The equation of the fitted line was found to be p1 = 1.091*p2 + 0.000. I.e the boundary that separates the two phases is where p2 is slighlty less than p1.

