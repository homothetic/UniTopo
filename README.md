# UniTopo
This repo is the official PyTorch implementation for paper: Unified Modeling of Lane and Lane Topology for Driving Scene Reasoning.

Autonomous vehicles need to perceive not only physical elements in the driving scene, such as lane lines and traffic lights, but also logical elements like lane centerlines and their topology.
Existing lane topology reasoning methods typically follow a reasoning-by-detection paradigm, where lane topological relationships are primarily derived from lane detection results.
In this paper, we propose an innovative method called Unified Modeling of Lane and Lane Topology (UniTopo), which represents the topological relationships between lanes as connected lanes, encompassing predecessor lanes, successor lanes, and their interconnections.
This unified representation of lanes and lane topology allows us to simultaneously obtain both the positions and topological information of lanes within a shared perception pipeline, establishing a new paradigm for directly perceiving lane topology from original image features.
We validate our method on the driving scene reasoning benchmark OpenLane-V2, which consists of two subsets, built based on Argoverse2 and nuScenes, respectively.
Our method achieves $\text{TOP}_{ll}$ of 30.1\% and 31.8\% on the two subsets, significantly surpassing the existing state-of-the-art method TopoFormer by 6.0\% and 8.6\%.
