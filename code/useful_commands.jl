#####################################################################
### Description: This file list commands to access memory/workers/gpus 
#####################################################################

import Pkg; Pkg.add("CUDA")
using CUDA, Distributed

### many details
versioninfo(verbose=true)

### Amount of memory (in MiB)
Sys.total_memory()/2^20 # total memory
Sys.free_memory()/2^20 # free memory

### Details on cpus
Sys.cpu_info()

### Number of cpus
cpus = length(Sys.cpu_info())

### Number of gpus
gpus = length(devices())

### cores per gpu
corespergpu = cpus/gpus

### Memory GPU specific
CUDA.memory_status()      

### Testing out going over memory limits
n = 10^6 #number of rows
m = 10^6 #number of columns

a = cu(rand(n,m)) # we get error OutOfMemoryError() and command is not executed at all
