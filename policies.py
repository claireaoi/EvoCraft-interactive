import torch
import torch.nn as nn
import math
import numpy as np
from vectors_to_blocks import isBlock, symmetriseX


class MLP(nn.Module):
    """
    MLP

    """

    def __init__(self, one_hot_dim, embedding_dim, dimension, n_layers, bias=True):
        super(MLP, self).__init__()
        self.input_dim = dimension  # Initial dimension, here 2 if in2D or 3 if in 3D
        self.one_hot_dim = one_hot_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        # self.embedding_dim_1=math.floor(embedding_dim/2)#if one layer more

        self.linear1 = nn.Linear(self.input_dim, self.embedding_dim, bias=bias)
        self.relu = nn.ReLU()
        if self.n_layers == 2:
            self.linear2 = nn.Linear(
                self.embedding_dim, self.one_hot_dim, bias=bias)
        else:
            self.linear2 = nn.Linear(
                self.embedding_dim, self.embedding_dim, bias=bias)
            self.linear3 = nn.Linear(
                self.embedding_dim, self.one_hot_dim, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src):
        """
        When input a 2D or 3D position, output a type of block, and if oriented=True an orientation for the block at this position.
        In case the block is air, it would be then empty.
        Input:
            src dimension: vector of dimension (1, input_space_dim), for instance 2 or 3, spatial coordinates
        Output:
            out: vector of probability, dimension (one_hot_dim) of having a block type as output
        """

        x1 = self.linear1(src)
        x1 = self.relu(x1)  # or sigmoid
        if self.n_layers == 2:
            out = self.linear2(x1)
            # out=x3[-1] #Last dimension, ie size one_hot_dim
            out[:, 0] = self.sigmoid(out[:, 0])
            out[:, 1:] = self.softmax(out[:, 1:])
        else:
            x2 = self.linear2(x1)
            x2 = self.sigmoid(x2)
            out = self.linear3(x2)
            # out=x3[-1] #Last dimension, ie size one_hot_dim
            out[:, 0] = self.sigmoid(out[:, 0])
            out[:, 1:] = self.softmax(out[:, 1:])
        return out[-1]

    def get_weights(self):
        param = nn.utils.parameters_to_vector(
            self.parameters()).detach().numpy()
        print("weights param size", param.shape)
        return param

    def generate_structure(self, generator_init_params):
        symmetrise = generator_init_params['symmetrise']
        dimension = generator_init_params['dimension']
        if dimension == 3:  # CASE 2 DIM
            blocks, orientations, matter_blocks = self.generate_structure_2D(
                generator_init_params)
        else:
            blocks, orientations, matter_blocks = self.generate_structure_3D(
                generator_init_params)
        if symmetrise:
            sym_blocks = symmetriseX(blocks)
            sym_orientations = symmetriseX(orientations)
            return sym_blocks, sym_orientations, 2*matter_blocks
        else:
            return blocks, orientations, matter_blocks

    def generate_structure_3D(self, generator_init_params):
        """
        Given a policy network (MLP), and certain parameters, generate a 3D structure with top-k-sampling.
        Inputs:
            p: policy network (MLP here)
            generator_init_params: dictionary of parameters given as input of train
        Outputs:
            blocks: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                blocks[x,y,z] is an index indicating which block type shall be found there. If -1, means it should be air.
            orientations: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                orientations[x,y,z] is an index between 0 to 5 indicating which orientations shall the block be oriented. 
                Only matter if take in account the orientations.

        """
        bounds = generator_init_params['bounds']
        oriented = generator_init_params['oriented']
        top_k = max(1, generator_init_params['top_k'])
        # indicate if build AIR block or not
        density_threshold = generator_init_params['density_threshold']
        # BUILD CREATURE look at all blocks within bounds
        # BY DEFAULT BLOCK TYPES ARE AIR, so EMPTY
        blocks = (-1) * np.ones((bounds[0], bounds[1], bounds[2]), dtype=int)
        orientations = np.zeros((bounds[0], bounds[1], bounds[2]), dtype=int)
        input_symmetry = generator_init_params['input_symmetry']
        max_radius = generator_init_params['max_radius']
        radial_bound = generator_init_params['radial_bound']
        matter_blocks = 0

        for x in range(bounds[0]):
            for y in range(bounds[1]):  # this is height in minecraft
                for z in range(bounds[2]):
                    # NORMALISE POSITION: Map [0, bound]  into [-1, 1]: -1 + 2 x /bound
                    position_norm = torch.tensor(
                        [-1 + 2 * x/bounds[0], -1 + 2*y/bounds[1], -1 + 2*z/bounds[2]])  # BETWEEN -1 and 1
                    # Take norm in 1 dimension
                    dist = torch.norm(position_norm, dim=0)
                    dist = torch.unsqueeze(dist, dim=0)
                    # Limit to a radius if decide so
                    if (radial_bound and dist < max_radius) or not radial_bound:
                        # only for x and z HERE ! but that may slow down..
                        if input_symmetry == 'sin':
                            position_norm = torch.tensor([torch.sin(
                                math.pi*position_norm[0]), position_norm[1], torch.sin(math.pi*position_norm[2])])
                        elif input_symmetry == 'abs':
                            position_norm = torch.abs(position_norm)
                        position_in = torch.cat([position_norm, dist], dim=0)
                        position_in = torch.unsqueeze(position_in, dim=0)
                        out_blocks = self.forward(position_in)
                        out_dim = out_blocks.shape[0]
                        # Split block types and orientations, if orientation
                        if oriented:
                            out_air, out_blocks, out_orientations = torch.split(
                                out_blocks, [1, out_dim-7, 6], dim=0)  # orientation is size 6
                        else:
                            out_air, out_blocks = torch.split(
                                out_blocks, [1, out_dim-1], dim=0)
                        air = bool(out_air.detach().numpy()
                                   [0] < density_threshold)
                        # FIRST DIMENSION indicate if shall build an AIR block or not
                        if not air:  # IF NOT AIR CHOOSE a block type and an orientation
                            matter_blocks += 1
                            # TOP K Sampling for blocks
                            top_out, top_index = torch.topk(
                                out_blocks, top_k, dim=0)
                            top_proba = nn.functional.softmax(top_out, dim=0)
                            block_index = np.random.choice(
                                top_index.detach().numpy(), 1, p=top_proba.detach().numpy())[0]
                            blocks[x, y, z] = int(block_index)
                            # Top K sampling for orientations
                            if oriented:
                                top_out_orient, top_index_orient = torch.topk(
                                    out_orientations, top_k, dim=0)
                                top_proba_orient = nn.functional.softmax(
                                    top_out_orient, dim=0)
                                orient_index = np.random.choice(top_index_orient.detach(
                                ).numpy(), 1, p=top_proba_orient.detach().numpy())[0]
                                orientations[x, y, z] = int(orient_index)
        return blocks, orientations, matter_blocks

    def generate_structure_2D(self, generator_init_params):
        """
        Given a policy network (MLP), and certain parameters, generate a 2D structure with top-k-sampling.
        Inputs:
            p: policy network (MLP here)
            generator_init_params: dictionary of parameters given as input of train
        Outputs:
            blocks: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                blocks[x,y] is an index indicating which block type shall be found there. If -1, means it should be air.
            orientations: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                orientations[x,y] is an index between 0 to 5 indicating which orientations shall the block be oriented. 
                Only matter if take in account the orientations.

        """
        bounds = generator_init_params['bounds']
        oriented = generator_init_params['oriented']
        top_k = max(1, generator_init_params['top_k'])
        # indicate if build AIR block or not
        density_threshold = generator_init_params['density_threshold']
        input_symmetry = generator_init_params['input_symmetry']
        max_radius = generator_init_params['max_radius']
        radial_bound = generator_init_params['radial_bound']

        # BUILD CREATURE look at all blocks within bounds
        # BY DEFAULT BLOCK TYPES ARE AIR, so EMPTY
        blocks = (-1) * np.ones((bounds[0], bounds[1]), dtype=int)
        orientations = np.zeros((bounds[0], bounds[1]), dtype=int)
        matter_blocks = 0
        for x in range(bounds[0]):
            for y in range(bounds[1]):  # this is height in minecraft

                # Map [0, bound]  into [-1, 1]: -1 + 2 x /bound
                position_norm = torch.tensor(
                    [-1 + 2 * x/bounds[0], -1 + 2*y/bounds[1]])
                # ADD distance to center to input as possibly input
                # Take norm in 0 dimension
                dist = torch.norm(position_norm, dim=0)
                dist = torch.unsqueeze(dist, dim=0)
                # Limit to a radius if decide so
                if (radial_bound and dist < max_radius) or not radial_bound:
                    # only for x and z HERE ! but that may slow down..
                    if input_symmetry == 'sin':
                        position_norm = torch.tensor(
                            [torch.sin(math.pi*position_norm[0]), position_norm[1]])
                    elif input_symmetry == 'abs':
                        position_norm = torch.abs(position_norm)
                    position_in = torch.cat([position_norm, dist], dim=0)
                    position_in = torch.unsqueeze(position_in, dim=0)
                    out_blocks = self.forward(position_in)
                    out_dim = out_blocks.shape[0]
                    # Split block types and orientations, if orientation
                    if oriented:
                        out_air, out_blocks, out_orientations = torch.split(
                            out_blocks, [1, out_dim-7, 6], dim=0)  # orientation is size 6
                    else:
                        out_air, out_blocks = torch.split(
                            out_blocks, [1, out_dim-1], dim=0)
                    air = bool(out_air.detach().numpy()[0] < density_threshold)
                    # FIRST DIMENSION indicate if shall build an AIR block or not
                    if not air:  # IF NOT AIR CHOOSE a block type and an orientation
                        matter_blocks += 1
                        # TOP K Sampling for blocks
                        top_out, top_index = torch.topk(
                            out_blocks, top_k, dim=0)
                        top_proba = nn.functional.softmax(top_out, dim=0)
                        block_index = np.random.choice(
                            top_index.detach().numpy(), 1, p=top_proba.detach().numpy())[0]
                        blocks[x, y] = int(block_index)
                        # Top K sampling for orientations
                        if oriented:
                            top_out_orient, top_index_orient = torch.topk(
                                out_orientations, top_k, dim=0)
                            top_proba_orient = nn.functional.softmax(
                                top_out_orient, dim=0)
                            orient_index = np.random.choice(top_index_orient.detach(
                            ).numpy(), 1, p=top_proba_orient.detach().numpy())[0]
                            orientations[x, y] = int(orient_index)

        return blocks, orientations, matter_blocks


class RNN(nn.Module):
    """
    RNN
    """

    def __init__(self, one_hot_dim, embedding_dim, hidden_dim, n_layers, if_embedding):
        super(RNN, self).__init__()

        self.one_hot_dim = one_hot_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers  # number of stacked RNN
        self.if_embedding = if_embedding
        if if_embedding:  # embedding if put two different size only for embedding etc
            self.embedding = nn.Embedding(one_hot_dim, embedding_dim)
            self.rnn = nn.RNN(embedding_dim, hidden_dim,
                              n_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(one_hot_dim, hidden_dim,
                              n_layers, batch_first=True)
        #self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, one_hot_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        INPUTS:
            input is the sequence which is fed into the network, of size (batch, seq_len, one_hot_dim) (since here batch_first=True)
            h_0 is the initial hidden state of the network, of size (num_layers, batch, hidden_size). 
        OUTPUTS of RNN:
            out is the output of the RNN from all timesteps from the last RNN layer, passed then by a linear Fc layer. It is of the size (batch, seq_len, one_hot_dim) (since here batch_first=True)
            h_n is the hidden value from the last time-step of all RNN layers, of size (num_layers, batch, hidden_size).

        """
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined above
        h0 = self.init_hidden(batch_size)
        # ONLY IF WANT AN EMBEDDING before hand
        if self.if_embedding:
            x = self.embedding(x)
        # Passing in the input and hidden state into the model and obtaining output
        # output of RNN is (batch, seq_len, hidden_dim)
        out, hidden = self.rnn(x, h0)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # SIZE (seq_len, hidden_dim) HERE 20 for now
        out = out.contiguous().view(-1, self.hidden_dim)
        out2 = self.fc(out)  # Size(seq_len, out_dim) HERE 13 for now
        out3 = self.softmax(out2)  # output proba for each elt dimension d...
        return out3, hidden

    def init_hidden(self, batch_size):
        # Generates the first hidden state of zeros h_0 for forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def get_weights(self):
        return nn.utils.parameters_to_vector(self.parameters()).detach().numpy()

    def generate_structure(self, generator_init_params, count_run):
        """
        Given a policy network (RNN), and certain parameters, generate a structure with a top-k-sampling.
        Inputs:
            p: policy network (RNN here)
            generator_init_params: dictionary of parameters given as input of train
        Outputs:
            blocks_indices: list of indices representing a Minecraft structure (in the chosen encoding). 
            These indices may to sos,eos tokens, directions, block type and possibly orientations)

        """
        # here policy already init with parameters">>>
        one_hot_dim = generator_init_params['output_dim']
        max_sequence_size = generator_init_params['max_sequence_size']
        top_k = max(1, generator_init_params['top_k'])
        min_size = generator_init_params['min_size']
        oriented = generator_init_params['oriented']
        graph = generator_init_params['graph']
        dimension = generator_init_params['dimension']

        # BUILD CREATURE STARTING FROM SOS with MAXIMUM SIZE, STOP IF ENCOUNTER EOS
        blocks = torch.zeros([1, 1, one_hot_dim])
        blocks[0, 0, 1] = 1  # SOS
        blocks_indices = [0]  # start with SOS, of indice 0
        eos = False
        t, n_blocks = 0, 0
        while not eos and t < max_sequence_size:
            # iterate, here blocks and out are of shape (1, t+1, d)
            out, hidden = self.forward(blocks)
            proba = out[t, :]  # this is proba
            # TOP K SAMPLING:
            top_out, top_index = torch.topk(proba, top_k, dim=0)
            top_proba = nn.functional.softmax(top_out, dim=0)
            new_block_index = np.random.choice(
                top_index.detach().numpy(), 1, p=top_proba.detach().numpy())[0]
            blocks_indices.append(new_block_index)
            new_block = torch.zeros([1, 1, one_hot_dim])
            new_block[0, 0, new_block_index] = 1
            # add 1 if the last index is a block material
            n_blocks += int(isBlock(new_block_index,
                                    oriented, graph, dimension))
            eos = bool(new_block_index == 1)  # THEN EOS so stop the creation
            # Concatenate along 1st dimension
            blocks = torch.cat((blocks, new_block), 1)
            t += 1
        # IN CASE EOS was GENERATED TOO EARLY, not mini size, regenerate.
        if n_blocks < min_size and count_run < 10 and top_k > 1:
            print("Regenerate, was too small")
            blocks_indices = self.generate_structure(
                generator_init_params, count_run+1)

        return blocks_indices


def gaussianFilter(x):
    return torch.exp((-1/2) * torch.square(x))


class SymMLP(nn.Module):
    """
    MLP with symmetric or periodic activations

    """

    def __init__(self, one_hot_dim, embedding_dim, dimension, bias=True):
        super(SymMLP, self).__init__()
        self.input_dim = dimension  # Initial dimension, here 2 if in2D or 3 if in 3D
        self.one_hot_dim = one_hot_dim
        self.embedding_dim = embedding_dim
        self.linear11 = nn.Linear(
            self.input_dim, self.embedding_dim, bias=bias)
        self.linear12 = nn.Linear(
            self.input_dim, self.embedding_dim, bias=bias)
        self.linear21 = nn.Linear(
            2 * self.embedding_dim, self.embedding_dim, bias=bias)
        self.linear22 = nn.Linear(
            2 * self.embedding_dim, self.embedding_dim, bias=bias)
        self.linear23 = nn.Linear(
            2 * self.embedding_dim, self.embedding_dim, bias=bias)
        self.linear3 = nn.Linear(
            3 * self.embedding_dim, self.one_hot_dim, bias=bias)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.sin = torch.sin
        self.tanh = torch.tanh
        self.cos = torch.cos
        self.gaussian = gaussianFilter

    def forward(self, src):
        """
        When input a 2D or 3D position, output a type of block, and if oriented=True an orientation for the block at this position.
        In case the block is air, it would be then empty.
        Input:
            src dimension: vector of dimension (1, input_space_dim), for instance 2 or 3, spatial coordinates.
            Beware, here coordinates have been normalised, son all between -1 and 1.

        Output:
            out: vector of probability, dimension (one_hot_dim) of having a block type as output
        """
        dim = src.shape[1]
        # dist = torch.norm(src, dim=1) #dist from center or add it before?

        # FIRST LAYER
        x11 = self.linear11(src)
        x12 = self.linear12(src)
        # ACTIVATIONS POST LAYER 1
        x11 = self.tanh(x11)
        x12 = self.gaussian(x12)  # or sigmoid?
        # SECOND LAYER
        x21 = self.linear21(torch.cat([x11, x12], dim=1))
        x22 = self.linear22(torch.cat([x11, x12], dim=1))
        x23 = self.linear23(torch.cat([x11, x12], dim=1))
        # ACTIVATIONS POST LAYER 2
        x21 = self.sin(x21)
        x22 = self.cos(x22)
        x23 = self.gaussian(x23)
        # THIRD LAYER
        x3 = self.linear3(torch.cat([x21, x22, x23], dim=1))

        # ACTIVATIONS POST LAYER 3
        out = self.gaussian(x3)  # or remove?

        # OUT
        # out=x3[-1] #Last dimension, ie size one_hot_dim
        out[:, 0] = self.sigmoid(out[:, 0])
        out[:, 1:] = self.softmax(out[:, 1:])
        return out[-1]

    def get_weights(self):
        param = nn.utils.parameters_to_vector(
            self.parameters()).detach().numpy()
        print("weights param size", param.shape)
        return param

    def generate_structure(self, generator_init_params):
        symmetrise = generator_init_params['symmetrise']
        dimension = generator_init_params['dimension']
        if dimension == 3:  # CASE 2 DIM
            blocks, orientations, matter_blocks = self.generate_structure_2D(
                generator_init_params)
        else:
            blocks, orientations, matter_blocks = self.generate_structure_3D(
                generator_init_params)
        if symmetrise:
            sym_blocks = symmetriseX(blocks)
            sym_orientations = symmetriseX(orientations)
            return sym_blocks, sym_orientations, 2*matter_blocks
        else:
            return blocks, orientations, matter_blocks

    def generate_structure_3D(self, generator_init_params):
        """
        Given a policy network (MLP), and certain parameters, generate a 3D structure with top-k-sampling.
        Inputs:
            p: policy network (MLP here)
            generator_init_params: dictionary of parameters given as input of train
        Outputs:
            blocks: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                blocks[x,y,z] is an index indicating which block type shall be found there. If -1, means it should be air.
            orientations: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                orientations[x,y,z] is an index between 0 to 5 indicating which orientations shall the block be oriented. 
                Only matter if take in account the orientations.

        """
        bounds = generator_init_params['bounds']
        oriented = generator_init_params['oriented']
        top_k = max(1, generator_init_params['top_k'])
        # indicate if build AIR block or not
        density_threshold = generator_init_params['density_threshold']
        # BUILD CREATURE look at all blocks within bounds
        # BY DEFAULT BLOCK TYPES ARE AIR, so EMPTY
        blocks = (-1) * np.ones((bounds[0], bounds[1], bounds[2]), dtype=int)
        orientations = np.zeros((bounds[0], bounds[1], bounds[2]), dtype=int)
        max_radius = generator_init_params['max_radius']
        radial_bound = generator_init_params['radial_bound']
        matter_blocks = 0

        for x in range(bounds[0]):
            for y in range(bounds[1]):  # this is height in minecraft
                for z in range(bounds[2]):
                    # NORMALISE POSITION: Map [0, bound]  into [-1, 1]: -1 + 2 x /bound
                    position_norm = torch.tensor(
                        [-1 + 2 * x/bounds[0], -1 + 2*y/bounds[1], -1 + 2*z/bounds[2]])  # BETWEEN -1 and 1
                    # Take norm in 1 dimension
                    dist = torch.norm(position_norm, dim=0)
                    dist = torch.unsqueeze(dist, dim=0)
                    # Limit to a radius if decide so
                    if (radial_bound and dist < max_radius) or not radial_bound:
                        position_in = torch.cat([position_norm, dist], dim=0)
                        position_in = torch.unsqueeze(position_in, dim=0)
                        out_blocks = self.forward(position_in)
                        out_dim = out_blocks.shape[0]
                        # Split block types and orientations, if orientation
                        if oriented:
                            out_air, out_blocks, out_orientations = torch.split(
                                out_blocks, [1, out_dim-7, 6], dim=0)  # orientation is size 6
                        else:
                            out_air, out_blocks = torch.split(
                                out_blocks, [1, out_dim-1], dim=0)
                        air = bool(out_air.detach().numpy()
                                   [0] < density_threshold)
                        # FIRST DIMENSION indicate if shall build an AIR block or not
                        if not air:  # IF NOT AIR CHOOSE a block type and an orientation
                            matter_blocks += 1
                            # TOP K Sampling for blocks
                            top_out, top_index = torch.topk(
                                out_blocks, top_k, dim=0)
                            top_proba = nn.functional.softmax(top_out, dim=0)
                            block_index = np.random.choice(
                                top_index.detach().numpy(), 1, p=top_proba.detach().numpy())[0]
                            blocks[x, y, z] = int(block_index)
                            # Top K sampling for orientations
                            if oriented:
                                top_out_orient, top_index_orient = torch.topk(
                                    out_orientations, top_k, dim=0)
                                top_proba_orient = nn.functional.softmax(
                                    top_out_orient, dim=0)
                                orient_index = np.random.choice(top_index_orient.detach(
                                ).numpy(), 1, p=top_proba_orient.detach().numpy())[0]
                                orientations[x, y, z] = int(orient_index)
        return blocks, orientations, matter_blocks

    def generate_structure_2D(self, generator_init_params):
        """
        Given a policy network (MLP), and certain parameters, generate a 2D structure with top-k-sampling.
        Inputs:
            p: policy network (MLP here)
            generator_init_params: dictionary of parameters given as input of train
        Outputs:
            blocks: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                blocks[x,y] is an index indicating which block type shall be found there. If -1, means it should be air.
            orientations: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                orientations[x,y] is an index between 0 to 5 indicating which orientations shall the block be oriented. 
                Only matter if take in account the orientations.

        """
        bounds = generator_init_params['bounds']
        oriented = generator_init_params['oriented']
        top_k = max(1, generator_init_params['top_k'])
        # indicate if build AIR block or not
        density_threshold = generator_init_params['density_threshold']
        max_radius = generator_init_params['max_radius']
        radial_bound = generator_init_params['radial_bound']

        # BUILD CREATURE look at all blocks within bounds
        # BY DEFAULT BLOCK TYPES ARE AIR, so EMPTY
        blocks = (-1) * np.ones((bounds[0], bounds[1]), dtype=int)
        orientations = np.zeros((bounds[0], bounds[1]), dtype=int)
        matter_blocks = 0
        for x in range(bounds[0]):
            for y in range(bounds[1]):  # this is height in minecraft

                # Map [0, bound]  into [-1, 1]: -1 + 2 x /bound
                position_norm = torch.tensor(
                    [-1 + 2 * x/bounds[0], -1 + 2*y/bounds[1]])
                # ADD distance to center to input as possibly input
                # Take norm in 0 dimension
                dist = torch.norm(position_norm, dim=0)
                dist = torch.unsqueeze(dist, dim=0)
                # Limit to a radius if decide so
                if (radial_bound and dist < max_radius) or not radial_bound:
                    position_in = torch.cat([position_norm, dist], dim=0)
                    position_in = torch.unsqueeze(position_in, dim=0)
                    out_blocks = self.forward(position_in)
                    out_dim = out_blocks.shape[0]
                    # Split block types and orientations, if orientation
                    if oriented:
                        out_air, out_blocks, out_orientations = torch.split(
                            out_blocks, [1, out_dim-7, 6], dim=0)  # orientation is size 6
                    else:
                        out_air, out_blocks = torch.split(
                            out_blocks, [1, out_dim-1], dim=0)
                    air = bool(out_air.detach().numpy()[0] < density_threshold)
                    # FIRST DIMENSION indicate if shall build an AIR block or not
                    if not air:  # IF NOT AIR CHOOSE a block type and an orientation
                        matter_blocks += 1
                        # TOP K Sampling for blocks
                        top_out, top_index = torch.topk(
                            out_blocks, top_k, dim=0)
                        top_proba = nn.functional.softmax(top_out, dim=0)
                        block_index = np.random.choice(
                            top_index.detach().numpy(), 1, p=top_proba.detach().numpy())[0]
                        blocks[x, y] = int(block_index)
                        # Top K sampling for orientations
                        if oriented:
                            top_out_orient, top_index_orient = torch.topk(
                                out_orientations, top_k, dim=0)
                            top_proba_orient = nn.functional.softmax(
                                top_out_orient, dim=0)
                            orient_index = np.random.choice(top_index_orient.detach(
                            ).numpy(), 1, p=top_proba_orient.detach().numpy())[0]
                            orientations[x, y] = int(orient_index)

        return blocks, orientations, matter_blocks


class CPPN(nn.Module):
    pass
