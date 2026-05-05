## Problem Description 
I'm working on drone based urban traffic forecasting.

I have a simulated urban traffic dataset, containing the traffic flow and density of all road segments in an urban road network.

The data sequences are organized as tensors of shape (S, T, P, C), where S is the number of simulation sessions, and T are the time steps (all simulation sessions lasts 2 hours), P is the number of road segments, and C is the feature dimension (including flow, density and optionally time-in-day encodings).

The urban region is divided into grids, a road segment belongs to a cell when its center falls into the cell. I have an adjacency matrix based on road network topology, if one road segment is physically linked to another, then they are connected in the graph representation.

Every time step (3 minutes), I can select a few grid cells to be simultaneously visible, and thereby having the traffic flow and density of all road segments belonging to the selected cells for this very time step (past 3 minutes to now).

There will be K initial cells (drone monitoring places) randomly chose, with corresponding initial traffic observations. The drones can move 1 grid step at a time. 

The agent is rewarded when it can observe the change of congestion state, from congested to uncongested or the opposite.

And I’m training an agent to do this via reinforcement learning, where I implemented my own environment with gymnasium and utilized the training pipeline of stable-baselines3.

