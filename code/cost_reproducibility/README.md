Time the computation with

- pip frozen + torch deterministic yes/no
- conda frozen
- guix 
- guix + AVX512 mask
- guix + AVX2 mask


# Launch experiments locally

To run locally, you can use the function written in setup_environment.sh in each directory which define a bash function launch_my_computation:

```
source setup_environment.sh
launch_my_computation ../results 
```

Then, you can find the result in the `results` folder.