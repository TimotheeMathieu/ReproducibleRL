# guix-rl

Guix channel for RL 

## Usage
Add the following either to you channels file or as an argument for your guix cli command.

file: channels.scm
```scheme
 (cons 
     (channel
        (name 'guix-rl)
        (url "https://gitlab.inria.fr/tmathieu/guix-rl")
        (branch "main"))
    %default-channels
        )
```

for example, can be used with `guix shell` with (here with -C for container and -N for network)

```
guix time-machine --channels=channels.scm -- shell -CN python python-gymnasium -- python
```


## Packages available in this channel

Here is the list of packages available in this channel (obtained through `guix package -A ".*" | grep "guix-rl"`).

```
gcvir               	0.00                	out,debug,static    	guix-rl/packages/tools.scm:47:2
marchingcubecpp     	git-20230911        	out                 	guix-rl/packages/tools.scm:138:2
mujoco              	3.3.1               	out                 	guix-rl/packages/reinforcement-learning.scm:59:3
python-ale-py       	0.10.2              	out                 	guix-rl/packages/reinforcement-learning.scm:122:2
python-box2d-py     	2.3.5               	out                 	guix-rl/packages/tools.scm:178:2
python-dm-env       	1.6                 	out                 	guix-rl/packages/tools.scm:213:2
python-gymnasium-ale-py	1.1.1               	out                 	guix-rl/packages/reinforcement-learning.scm:226:2
python-gymnasium-all	1.1.1               	out                 	guix-rl/packages/reinforcement-learning.scm:260:2
python-gymnasium-box2d	1.1.1               	out                 	guix-rl/packages/reinforcement-learning.scm:289:2
python-gymnasium-classic-control	1.1.1               	out                 	guix-rl/packages/reinforcement-learning.scm:322:2
python-gymnasium-mujoco	1.1.1               	out                 	guix-rl/packages/reinforcement-learning.scm:354:2
python-gymnasium-next	1.1.1               	out                 	guix-rl/packages/reinforcement-learning.scm:386:2
python-minatar      	1.0.15              	out                 	guix-rl/packages/reinforcement-learning.scm:420:2
python-moviepy      	2.2.1               	out                 	guix-rl/packages/tools.scm:256:2
python-mujoco       	3.3.1               	out                 	guix-rl/packages/reinforcement-learning.scm:460:2
python-proglog      	0.1.12              	out                 	guix-rl/packages/tools.scm:292:2
python-pytinyrenderer	0.0.14              	out                 	guix-rl/packages/tools.scm:315:2
python-stable-baselines3	2.6.0               	out                 	guix-rl/packages/reinforcement-learning.scm:498:2
sdflib              	git-1927bee6bb8225258a39c8cbf14e18a4d50409ae	out                 	guix-rl/packages/tools.scm:333:2
tinyobjloader       	git-1421a10d6ed9742f5b2c1766d22faa6cfbc56248	out                 	guix-rl/packages/tools.scm:418:2
```

Remark: gcvir is an implementation of libvircpuid meant to be grafted in place of glibc in order to mask AVX512 flag (needed for better reproducibility).