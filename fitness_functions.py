import numpy as np
import torch
import torch.nn as nn
from typing import List, Any
import sys

from policies import RNN, MLP, SymMLP
from vectors_to_blocks import *


def getch():
    """ Allows to input values without pressing enter """
    import termios
    import sys
    import tty

    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch()



def fitness_MLP(evolved_parameters: np.array, generator_init_params: dict, restricted_flag: bool) -> float:
    """ 
    For an interactive evolution, where each structure is presented to the human.
    Query the fitness of a structure generated by a given policy network (MLP), whose parameters are given in argument.
    input:
        evolved_parameters: parameters of a policy network p, element of the population
        generator_init_params: dictionary of parameters given as input of train
        restricted_flag: boolean, tells if work with a restricted number of blocks
    output:
        reward for the network p

    """
    # Unload generator init parameters
    one_hot_dim = generator_init_params['output_dim']
    embedding_dim = generator_init_params['embedding_dim']
    offset = generator_init_params['position']
    oriented = generator_init_params['oriented']
    dimension = generator_init_params['dimension']
    n_layers = generator_init_params['n_layers']
    bounds = generator_init_params['bounds']
    symmetrise = generator_init_params['symmetrise']
    min_size = generator_init_params['min_size']

    if symmetrise:
        neo_bound = 2*bounds[0]
    else:
        neo_bound = bounds[0]
    # Initilise policy network
    p = MLP(one_hot_dim, embedding_dim, dimension, n_layers)
    # Load weights into the policy network
    nn.utils.vector_to_parameters(torch.tensor(
        evolved_parameters, dtype=torch.float32),  p.parameters())
    # Generate structure with MLP
    blocks, orientations, matter_blocks = p.generate_structure(
        generator_init_params)

    if matter_blocks > min_size:

        # Build it in MInecraft
        build_zone(blocks, offset, restricted_flag,
                   orientations, oriented, dimension)
        # Ask Rating Human
        print("Rate the creation from 1 to 5:")
        reward = float(getch())
        print(reward)
        # Clean blocks function afterwards: clean all zone
        clean_zone([neo_bound, bounds[1], bounds[2]], offset)
    else:
        reward = 0
    scaled_reward = float(max(min(reward, 5), 1)*10)
    return scaled_reward  # bound reward from 10 to 50


def fitness_MLP_alltogether(evolved_parameters: np.array, generator_init_params: dict, restricted_flag: bool) -> float:
    """
    For an interactive evolution, with N(=batch_size) structures compared.
    Ask the human to select a structure among N structures generated each by a different policy network (MLP).
    input:
        evolved_parameters: np.array of size batch_size * N_policy_param(>>), gathering the parameters of the batch_size policy network compared here.
        generator_init_params: dictionary of parameters given as input of train
        restricted_flag: boolean, tells if work with a restricted number of blocks
    output:
        rewards: list of size batch_size of rewards for each of these networks.

    """
    # Unload generator init parameters
    one_hot_dim = generator_init_params['output_dim']
    embedding_dim = generator_init_params['embedding_dim']
    origin = generator_init_params['position']
    batch_size = generator_init_params['choice_batch']
    bounds = generator_init_params['bounds']
    oriented = generator_init_params['oriented']
    dimension = generator_init_params['dimension']
    population_size = generator_init_params['population_size']
    min_size = generator_init_params['min_size']
    symmetrise = generator_init_params['symmetrise']
    n_layers = generator_init_params['n_layers']
    global_count = 0
    local_count = 0
    # INITIALISATION of some parameters
    blocks_batch = []
    spacing = 7  # spacing between 2 structure
    # If symmetric structure, like if bound[0] get multiplied by 2
    if symmetrise:
        neo_bound = 2*bounds[0]
    else:
        neo_bound = bounds[0]

    width_batch = (neo_bound + spacing) * batch_size
    # so player can see the different creatures
    perspective = np.floor(width_batch/2)
    # print("Position player suggested:", origin)
    rewards = [0]*population_size
    batch_indices = []

    # Loop on the whole structure
    while global_count < population_size:
        while local_count < batch_size and global_count < population_size:
            # Initilise new policy network
            p = MLP(one_hot_dim, embedding_dim, dimension, n_layers)
            # Load weights into the policy network
            nn.utils.vector_to_parameters(torch.tensor(
                evolved_parameters[global_count, :], dtype=torch.float32),  p.parameters())
            # Generate structure with MLP
            blocks, orientations, matter_blocks = p.generate_structure(
                generator_init_params)
            global_count += 1
            if matter_blocks > min_size:  # Then non empty structure, and structure big enough
                blocks_batch.append(blocks)
                local_count += 1
                batch_indices.append(global_count-1)
        # Build in Minecraft these different structures aligned
        # AS now, never rate the last incomplete batch
        if local_count == batch_size:
            offset = [origin[0] -
                      np.floor(width_batch/2), 4, origin[2] - perspective]
            # print("Offset", offset)
            for i in range(len(blocks_batch)):
                build_zone(
                    blocks_batch[i], offset, restricted_flag, orientations, oriented, dimension)
                # redefine first coordinate offset
                offset[0] += neo_bound + spacing
            # Ask Rating Human
            print("Choose one of these structures, from 1 to " +
                  str(batch_size) + " (west to east) ")

            while True:
                try:
                    index = int(getch())
                    if index not in np.arange(len(batch_indices))+1:
                        print('Entry not valid, try again. Press 0 to EXIT')
                    else:
                        break
                except:
                    print('Entry not valid, try again')
                if index == 0:
                    offset = [
                        origin[0] - np.floor(width_batch/2), 4, origin[2] - perspective]
                    full_bounds = [(neo_bound + spacing) *
                                   batch_size, bounds[1], bounds[2]]
                    clean_zone(full_bounds, offset)
                    sys.exit("Bye")

            # Update rewards of the picked creature
            selected_structure = batch_indices[index-1]
            # structure batch_size (last one) is global_count-1
            rewards[selected_structure] = 50
            # CHECK right structure reward:
            #print("current_indices:", batch_indices)
            #print("selected indices:", selected_structure)
            #print("selected blocks", all_blocks[selected_structure])
            # RESET variables for after and clean zone
            offset = [origin[0] -
                      np.floor(width_batch/2), 4, origin[2] - perspective]
            full_bounds = [(neo_bound + spacing) *
                           batch_size, bounds[1], bounds[2]]
            clean_zone(full_bounds, offset)
            local_count = 0
            blocks_batch = []
            batch_indices = []

    return rewards  # bound reward from 10 to 50 ?


def fitness_SymMLP(evolved_parameters: np.array, generator_init_params: dict, restricted_flag: bool) -> float:
    """ 
    For an interactive evolution, where each structure is presented to the human.
    Query the fitness of a structure generated by a given policy network (MLP), whose parameters are given in argument.
    input:
        evolved_parameters: parameters of a policy network p, element of the population
        generator_init_params: dictionary of parameters given as input of train
        restricted_flag: boolean, tells if work with a restricted number of blocks
    output:
        reward for the network p

    """
    # Unload generator init parameters
    one_hot_dim = generator_init_params['output_dim']
    embedding_dim = generator_init_params['embedding_dim']
    offset = generator_init_params['position']
    oriented = generator_init_params['oriented']
    dimension = generator_init_params['dimension']
    bounds = generator_init_params['bounds']
    symmetrise = generator_init_params['symmetrise']
    min_size = generator_init_params['min_size']

    if symmetrise:
        neo_bound = 2*bounds[0]
    else:
        neo_bound = bounds[0]
    # Initilise policy network
    p = SymMLP(one_hot_dim, embedding_dim, dimension)
    # Load weights into the policy network
    nn.utils.vector_to_parameters(torch.tensor(
        evolved_parameters, dtype=torch.float32),  p.parameters())
    # Generate structure with MLP
    blocks, orientations, matter_blocks = p.generate_structure(
        generator_init_params)

    if matter_blocks > min_size:
        # Build it in MInecraft
        build_zone(blocks, offset, restricted_flag,
                   orientations, oriented, dimension)
        # Ask Rating Human
        print("Rate the creation from 1 to 5:")
        reward = float(getch())
        print(reward)
        # Clean blocks function afterwards: clean all zone
        clean_zone([neo_bound, bounds[1], bounds[2]], offset)
    else:
        reward = 0
    scaled_reward = float(max(min(reward, 5), 1)*10)
    print("scaled rewards", scaled_reward)
    return scaled_reward  # bound reward from 10 to 50


def fitness_SymMLP_alltogether(evolved_parameters: np.array, generator_init_params: dict, restricted_flag: bool) -> float:
    """
    For an interactive evolution, with N(=batch_size) structures compared.
    Ask the human to select a structure among N structures generated each by a different policy network (MLP).
    input:
        evolved_parameters: np.array of size batch_size * N_policy_param(>>), gathering the parameters of the batch_size policy network compared here.
        generator_init_params: dictionary of parameters given as input of train
        restricted_flag: boolean, tells if work with a restricted number of blocks
    output:
        rewards: list of size batch_size of rewards for each of these networks.

    """
    # Unload generator init parameters
    one_hot_dim = generator_init_params['output_dim']
    embedding_dim = generator_init_params['embedding_dim']
    origin = generator_init_params['position']
    batch_size = generator_init_params['choice_batch']
    bounds = generator_init_params['bounds']
    oriented = generator_init_params['oriented']
    dimension = generator_init_params['dimension']
    population_size = generator_init_params['population_size']
    min_size = generator_init_params['min_size']
    symmetrise = generator_init_params['symmetrise']
    global_count = 0
    local_count = 0
    # INITIALISATION of some parameters
    blocks_batch = []
    spacing = 7  # spacing between 2 structure
    # If symmetric structure, like if bound[0] get multiplied by 2
    if symmetrise:
        neo_bound = 2*bounds[0]
    else:
        neo_bound = bounds[0]

    width_batch = (neo_bound + spacing) * batch_size
    # so player can see the different creatures
    perspective = np.floor(width_batch/2)
    # print("Position player suggested:", origin)
    rewards = [0]*population_size
    batch_indices = []

    # Loop on the whole structure
    while global_count < population_size:
        while local_count < batch_size and global_count < population_size:
            # Initilise new policy network
            p = SymMLP(one_hot_dim, embedding_dim, dimension)
            # Load weights into the policy network
            nn.utils.vector_to_parameters(torch.tensor(
                evolved_parameters[global_count, :], dtype=torch.float32),  p.parameters())
            # Generate structure with MLP
            blocks, orientations, matter_blocks = p.generate_structure(
                generator_init_params)
            global_count += 1
            if matter_blocks > min_size:  # Then non empty structure, and structure big enough
                blocks_batch.append(blocks)
                local_count += 1
                batch_indices.append(global_count-1)
        # Build in Minecraft these different structures aligned
        # AS now, never rate the last incomplete batch
        if local_count == batch_size:
            # initial offset #ON THE FLOOR OR PLAYER POSITION ?
            offset = [origin[0] -
                      np.floor(width_batch/2), 4, origin[2] - perspective]
            # print("Offset", offset)
            for i in range(len(blocks_batch)):
                build_zone(
                    blocks_batch[i], offset, restricted_flag, orientations, oriented, dimension)
                # redefine first coordinate offset
                offset[0] += neo_bound + spacing
            # Ask Rating Human
            print("Choose one of these structures, from 1 to " +
                  str(batch_size) + " (west to east) ")

            while True: 
                try:
                    index = int(getch())
                    if index not in np.arange(len(batch_indices))+1:
                        print('Entry not valid, try again. Press 0 to EXIT')
                    else:
                        break
                except:
                    print('Entry not valid, try again')
                if index == 0:
                    offset = [
                        origin[0] - np.floor(width_batch/2), 4, origin[2] - perspective]
                    full_bounds = [(neo_bound + spacing) *
                                   batch_size, bounds[1], bounds[2]]
                    # CLAIRE: TODO: make clean function self-contained
                    clean_zone(full_bounds, offset)
                    sys.exit("Bye")

            # Update rewards of the picked creature
            selected_structure = batch_indices[index-1]
            # structure batch_size (last one) is global_count-1
            rewards[selected_structure] = 50
            # RESET variables for after and clean zone
            offset = [origin[0] -
                      np.floor(width_batch/2), 4, origin[2] - perspective]
            full_bounds = [(neo_bound + spacing) *
                           batch_size, bounds[1], bounds[2]]
            clean_zone(full_bounds, offset)
            local_count = 0
            blocks_batch = []
            batch_indices = []

    return rewards  # bound reward from 10 to 50 ?


def fitness_RNN(evolved_parameters: np.array, generator_init_params: dict, restricted_flag: bool) -> float:
    """ 
    For an interactive evolution, where each structure is presented to the human.
    Query the fitness of a structure generated by a given policy network (RNN), whose parameters are given in argument.
    input:
        evolved_parameters: parameters of a policy network p (RNN), element of the population
        generator_init_params: dictionary of parameters given as input of train
        restricted_flag: boolean, tells if work with a restricted number of blocks
    output:
        reward for the network p
    """

    # Unload generator init parameters
    one_hot_dim = generator_init_params['output_dim']
    embedding_dim = generator_init_params['embedding_dim']
    hidden_dim = generator_init_params['hidden_dim_RNN']
    n_layers = generator_init_params['n_layers_RNN']
    if_embedding = generator_init_params['if_embedding']
    oriented = generator_init_params['oriented']
    offset = generator_init_params['position']
    graph = generator_init_params['graph']
    dimension = generator_init_params['dimension']
    min_size = generator_init_params['min_size']

    # Initilise policy network
    p = RNN(one_hot_dim, embedding_dim, hidden_dim, n_layers, if_embedding)
    # Load weights into the policy network
    nn.utils.vector_to_parameters(torch.tensor(
        evolved_parameters, dtype=torch.float32),  p.parameters())
    # BUILD CREATURE STARTING FROM SOS with MAXIMUM SIZE, STOP IF ENCOUNTER EOS
    blocks_indices = p.generate_structure(generator_init_params, 0)
    phenotype, blocks, positions, orientations = sequence_to_construct(blocks_indices, [], [], [], [], [
                                                                       [0, 0, 0]], [1], [1], restricted_flag, oriented, graph, dimension)  # default direction is north ?
    matter_blocks = len(blocks)
    if matter_blocks > min_size:
        # print("Constructible Sequence:", phenotype)
        positions = np.array(positions)
        tiled_offset = np.tile(offset, (len(blocks), 1))
        new_positions = np.add(positions, tiled_offset)  # np.tile
        build_from_sequence(blocks, new_positions, orientations)
        bounds_structure = [np.amin(np.array(positions), axis=0), np.amax(
            np.array(positions), axis=0)]

        print("Rate the creation from 1 to 5:")
        reward = float(getch())
        print(reward)
        if len(positions) > 0:
            clean_positions(new_positions)

        print("***** Thanks for helping me learning ****")
        # reward = float(input("Rate the creation from 1 to 5: "))
        # Add clean blocks function here

        # bound reward from 10 to 50 has to retuzrn list now
        return max(min(reward, 5), 1)*10
    else:
        return 0


def fitness_RNN_alltogether(evolved_parameters: np.array, generator_init_params: dict, restricted_flag: bool) -> float:
    """
    For an interactive evolution, with N(=batch_size) structures compared batch after batch.
    Ask the human to select a structure among N structures generated each by a different policy network (RNN).
    If the structure is empty, will remove it from the proposition
    input:
        evolved_parameters: np.array of size batch_size * N_policy_param(>>), gathering the parameters of the batch_size policy network compared here.
        generator_init_params: dictionary of parameters given as input of train
        restricted_flag: boolean, tells if work with a restricted number of blocks
    output:
        rewards: list of size batch_size of rewards for each of these networks.

    """

    # Unload generator init parameters
    one_hot_dim = generator_init_params['output_dim']
    embedding_dim = generator_init_params['embedding_dim']
    hidden_dim = generator_init_params['hidden_dim_RNN']
    n_layers = generator_init_params['n_layers_RNN']
    if_embedding = generator_init_params['if_embedding']
    oriented = generator_init_params['oriented']
    # SUGGEST PLAYER BE THERE TO SEE CREATURE
    origin = generator_init_params['position']
    batch_size = generator_init_params['choice_batch']
    graph = generator_init_params['graph']
    dimension = generator_init_params['dimension']
    population_size = generator_init_params['population_size']
    min_size = generator_init_params['min_size']

    # CREATURES WOULD BE ALIGNED ON WEST_EAST AXIS (IE FIRST COORDINATE), FROM WEST TO EAST,
    positions_batch, orientations_batch, blocks_batch, size_batch, batch_indices, new_positions_batch = [], [], [], [], [], []
    local_count, global_count, width_batch, spacing = 0, 0, 0, 5
    rewards = [0]*population_size

    # Loop on the whole structure
    while global_count < population_size:
        while local_count < batch_size and global_count < population_size:
            # Initilise policy network
            p = RNN(one_hot_dim, embedding_dim,
                    hidden_dim, n_layers, if_embedding)
            # Load weights into the policy network
            nn.utils.vector_to_parameters(torch.tensor(
                evolved_parameters[global_count, :], dtype=torch.float32),  p.parameters())
            # GENERATE STRUCTURE RNN
            blocks_indices = p.generate_structure(generator_init_params, 0)
            phenotype, blocks, positions, orientations = sequence_to_construct(blocks_indices, [], [], [], [],  [
                                                                               [0, 0, 0]], [1], [1], restricted_flag, oriented, graph, dimension)  # default direction is north ?
            matter_blocks = len(blocks)
            positions = np.array(positions)  # convert into np array
            assert(positions.shape[0] == len(blocks))
            global_count += 1

            if matter_blocks > min_size:  # Then non empty structure, and structure big enough
                # print("Constructible Sequence:", phenotype)
                local_count += 1
                batch_indices.append(global_count-1)
                positions_batch.append(positions)
                orientations_batch.append(orientations)
                blocks_batch.append(blocks)
                bounds_structure = [
                    np.amax(positions, axis=0), np.amin(positions, axis=0)]
                # only along one dimension
                size_structure = [bounds_structure[0]
                                  [0], bounds_structure[1][0]]
                size_batch.append(size_structure)
                width_batch += spacing + \
                    bounds_structure[0][0]-bounds_structure[1][0]  # width

        # Build in Minecraft these different structures aligned
        # AS now, never rate the last incomplete batch
        if local_count == batch_size:
            # so player can see the different creatures
            perspective = np.floor(width_batch/2)
            # initial offset #ON THE FLOOR OR PLAYER POSITION ?
            offset = [origin[0] -
                      np.floor(width_batch/2), 0, origin[2] - perspective]
            # CONSTRUCT AND PLACE STRUCTURES
            for i in range(len(blocks_batch)):
                # redefine first coordinate offset, according dimension
                offset[0] += spacing - size_batch[i][1]
                if i > 0:
                    offset[0] += size_batch[i-1][0]
                # UPDATE POSITION depending on offset
                tiled_offset = np.tile(offset, (len(blocks_batch[i]), 1))
                new_positions = np.add(
                    positions_batch[i], tiled_offset)  # np.tile
                new_positions_batch.append(new_positions)
                build_from_sequence(
                    blocks_batch[i], new_positions, orientations_batch[i])
            print("Choose one of these structures, from 1 to " +
                  str(batch_size) + " (west to east) ")

            while True:  
                try:
                    index = int(getch())
                    if index not in np.arange(len(batch_indices))+1:
                        print('Entry not valid, try again. Press 0 to EXIT')
                    else:
                        break
                except:
                    print('Entry not valid, try again')
                if index == 0:
                    clean_batch(new_positions_batch)
                    sys.exit("Bye")

            selected_structure = batch_indices[index-1]
            # structure batch_size (last one) is global_count-1
            rewards[selected_structure] = 50
            # ERASE THEN BOARD
            clean_batch(new_positions_batch)
            # RESET VARIABLES
            local_count = 0
            positions_batch, orientations_batch, blocks_batch, size_batch, batch_indices, new_positions_batch = [], [], [], [], [], []

    return rewards

