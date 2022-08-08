# 3DNester

Packing of 3D models for powder bed 3D printing. Also called nesting or 3D nesting. 
This project uses a simulated annealing algorithm for approach of a solution to the irregular 3D bin-packing problem. It currently supports the packing of one 3D model to maximum potential quantity. Octrees are used for collision detection in a broad and narrow phase detection algorithm.

## Further development

This repo was primarily used as a launchpad for quick development before moving into C++. The code was re-factored, and is currently being developed in my other repository: [3DNester-C++](https://github.com/ddm-j/3DNester-cpp)
Several algorithmic differences exist between this Python version and it's C++ equivalent. 
