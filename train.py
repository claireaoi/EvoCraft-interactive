import time
import argparse
import sys
import torch
from os.path import join, exists
from os import mkdir
import numpy as np

from evolution_strategy_static import EvolutionStrategyStatic
from policies import RNN, MLP, SymMLP
from vectors_to_blocks import RESTRICTED_BLOCKS, REMOVED_INDICES


# seed = 26
# torch.set_num_threads(1)
# torch.manual_seed(seed)
# np.random.seed(seed)


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--generator', type=str, default='MLP',
                        metavar='', help='Generator/policy type: MLP, RNN, SymMLP')
    parser.add_argument('--restricted', type=int, default=1, metavar='',
                        help='0 or 1. Tells if want to use only a restricted list of blocks (1), else 0.')
    parser.add_argument('--dimension', type=int, default=3, metavar='',
                        help='2 or 3. To restrict the spatial dimension we work with. ')
    parser.add_argument('--choice_batch', type=int, default=2, metavar='',
                        help='Number of structures among which to choose one.')
    parser.add_argument('--oriented', type=int, default=1, metavar='',
                        help='0 or 1. Indicate if shall incorporate the orientations in the encoding.')
    parser.add_argument('--position', type=list, default=[0, 10, 0], metavar='',
                        help='Initial position for player advised, around which the structures will be evolved.')
    parser.add_argument('--lr', type=float,  default=0.1,
                        metavar='', help='ES learning rate.')
    parser.add_argument('--decay', type=float,  default=0.99, metavar='',
                        help='ES and learning rate decay.') 
    parser.add_argument('--sigma', type=float,  default=0.4, metavar='',
                        help='ES sigma: modulates the amount of noise used to populate each new generation, the higher the more the entities will vary')
    parser.add_argument('--generations', type=int, default=30,
                        metavar='', help='Number of generations that the ES will run.')
    parser.add_argument('--population_size', type=int, default=2, metavar='',
                        help='Size of population (needs to be pair and be a multiple of choice_batch or will be approximated).')
    parser.add_argument('--top_k', type=int, default=1, metavar='',
                        help='Top-k sampling, for a stochastic generation of structures. For the deterministic case, choose k=1.')
    parser.add_argument('--folder', type=str, default='weights',
                        metavar='', help='folder to store the evolved weights ')


    args = parser.parse_args()

    assert args.dimension == 2 or args.dimension == 3
    assert args.choice_batch <= args.population_size

    if not exists(args.folder):
        mkdir(args.folder)

    # Initialise generator network and create constructor dictionary

    if args.generator == 'MLP':
        if args.restricted == 1:
            output_dim = int(len(RESTRICTED_BLOCKS) + 6*args.oriented+1)
        else:
            output_dim = int(254-len(REMOVED_INDICES)+6*args.oriented)
        generator_init_params = {
            'output_dim': output_dim,
            'embedding_dim': 50,
            'dimension': args.dimension+1,
            'restricted_encoding': args.restricted,
            # Bounds within which will query the network to build a structure. If 2D, the 3rd dimension will be ignored.
            'bounds': [6, 10, 10],
            # If query blocks only within a radial bound.
            'radial_bound': True,
            # If radial_bound=true, will inquire only blocks within a certain radius. Has to be between 0 and 1. Radius computed then depending on above bounds.
            'max_radius': 0.9,
            'top_k': args.top_k,
            'min_size': 3,  # Minimum number of block for a structure to proposed under the human rating
            'position': args.position,
            'choice_batch': args.choice_batch,
            'n_layers': 3,  # Number layers of the MLP. May be 2 or 3.
            'population_size': args.population_size,
            'oriented': bool(args.oriented),
            # Enable to have some control over the density of the structure.
            'density_threshold': 0.75,
            # May filter the input through 'sin' or 'abs' to enforce certain symmetry or regularities.
            'input_symmetry': 'abs',
            # If True, will symmetrise structure on the X axis.
            'symmetrise': False
        }
        p = MLP(generator_init_params['output_dim'], generator_init_params['embedding_dim'],
                generator_init_params['dimension'], generator_init_params['n_layers'])

    elif args.generator == 'SymMLP':
        if args.restricted == 1:
            output_dim = int(len(RESTRICTED_BLOCKS) + 6*args.oriented+1)
        else:
            output_dim = int(254-len(REMOVED_INDICES)+6*args.oriented)
        # Parameters for SymMLP are similar than for MLP.
        generator_init_params = {
            'output_dim': output_dim,
            # Note the embedding of the last layer will reach 3* embedding_dimension.
            'embedding_dim': 7,
            # Dimension of the structure + add distance to center
            'dimension': args.dimension+1,
            'restricted_encoding': args.restricted,
            'bounds': [10, 14, 10],
            'radial_bound': False,
            # if radial_bound=true, will inquire only blocks within a certain radius.BETWEEN 0 and 1 (will be comparatively to bounds above)
            'max_radius': 0.9,
            'top_k': args.top_k,
            # Below this size, will not show the structure to rate for the humam (and will have lowest possible reward)
            'min_size': 3,
            'position': args.position,
            'choice_batch': args.choice_batch,
            'population_size': args.population_size,
            'oriented': bool(args.oriented),
            # May enable to have some control over the density of the structure.>>> 
            'density_threshold': 0.7,
            'symmetrise': False  # decide if impose a symmetry on x when build structure
        }  # ie if first dimension MLP <0.3, then build AIR...
        p = SymMLP(generator_init_params['output_dim'],
                   generator_init_params['embedding_dim'], generator_init_params['dimension'])

    elif args.generator == 'RNN':
        graph_representation = 1  # if use graph encoding
        if args.restricted == 1:
            output_dim = int(len(RESTRICTED_BLOCKS)+2*args.dimension +
                             2+6*args.oriented + 2 * graph_representation)
        else:
            output_dim = int(254-len(REMOVED_INDICES)+2*args.dimension +
                             2+6*args.oriented + 2 * graph_representation)
        generator_init_params = {
            # Dimension of the output (depends on encoding, and on choice of including orientations or not)
            'output_dim': output_dim,
            'hidden_dim_RNN': 20,
            'dimension': args.dimension,  # Dimension of the structure
            'n_layers_RNN': 2,
            # If used a prior embedding instead of the one-hot representation for the RNN
            'if_embedding': False,
            'embedding_dim': 20,
            # Is used a restricted number of blocks
            'restricted_encoding': args.restricted,
            # Maximum length of the sequence when query the RNN... Or until EOS 
            'max_sequence_size': 80,
            'top_k': args.top_k,
            'oriented': bool(args.oriented),
            'position': args.position,
            'choice_batch': args.choice_batch,
            'min_size': 6,
            'population_size': args.population_size,
            # If choose encoding as a graph, i.e. with branching tokens in the sequence
            'graph': bool(graph_representation == 1)
        }
        p = RNN(generator_init_params['output_dim'], generator_init_params['embedding_dim'],
                generator_init_params['hidden_dim_RNN'], generator_init_params['n_layers_RNN'], generator_init_params['if_embedding'])

    else:
        raise NotImplementedError

    # Initialise the EvolutionStrategy class
    print('\nInitilisating ES for ' + args.generator)
    es = EvolutionStrategyStatic(p.get_weights(), generator=args.generator, generator_init_params=generator_init_params,
                                 restricted=args.restricted, sigma=args.sigma, learning_rate=args.lr, decay=args.decay)

    # Start the evolution
    es.run(args.generations, path=args.folder)


if __name__ == '__main__':
    main(sys.argv)
