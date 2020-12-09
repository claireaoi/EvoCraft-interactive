 
---

<div align="center">    
 
# Interactive evolution for EvoCraft
### Evolve intearctively Minecraft entities encoded as neural networks
###### Disclamer: This is an alpha version, expect further changes.
</div>
<br/>
<p align="center">
  <img src="example.gif">
</p>  

## Install

1. Set up the [python API to Minecraft](https://github.com/real-itu/Evocraft-py)

2. Clone the repo: `git clone https://github.com/claireaoi/EvoCraft-interactive`

3. Install dependencies: `pip install -r requirements.txt`


## Start interactive evolution

After [launching the Minecraft server](https://github.com/real-itu/Evocraft-py#2-starting-the-modded-minecraft-server) use `train.py` to start interactive evolution.

They generated machines will be rendered on Minecraft and you'll have to pick with the keyboard numbers which 
one you like most, this will provide a reward signal for the selected genotype. Press 0 to abort evolution.  
Evolution stops when the maximum generation number is reached.
The networks' weights (ie. the genotypes) will be saved in the `weights` folder. The evolution is guided by an [ES algorithm](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/).



Use `python train.py --help` to display all the evolution options:

```

train.py [--generator] [--restricted] [--dimension] [--choice_batch] [--oriented] [--position] [--lr] [--decay] [--sigma] [--generations] [--population_size] [--top_k] [--folder]

  --generator         Generator/policy type: MLP, RNN, SymMLP
  --restricted        0 or 1. Tells if want to use only a restricted list of blocks (1), else 0.
  --dimension         2 or 3. To restrict the spatial dimension we work with.
  --choice_batch      Number of structures among which to choose one.
  --oriented          0 or 1. Indicate if shall incorporate the orientations in the encoding.
  --position          Initial position for player advised, around which the structures will be evolved.
  --lr                ES learning rate.
  --decay             ES and learning rate decay.
  --sigma             ES sigma: modulates the amount of noise used to populate each new generation, the higher the more the entities will vary
  --generations       Number of generations that the ES will run.
  --population_size   Size of population (needs to be pair and be a multiple of choice_batch or will be approximated).
  --top_k             Top-k sampling, for a stochastic generation of structures. For the deterministic case, choose k=1.
  --folder            folder to store the evolved weights

```


<!-- ## Citation   

If you use the code for academic or commecial use, please cite the associated paper:

```bibtex

@article{
}

```    -->

