import numpy as np
import torch
import torch.nn as nn

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

from pandas.util import hash_array

class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim+self.memory_dim//4), dtype=torch.long), requires_grad=False)

        self.__init_memory_bank__()


    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()

    def hash_array(self, node_ids, seed):
        return (node_ids * (seed % 100) + node_ids^3 * ((seed % 100) + 1) + node_ids^5 * ((seed % 100) + 3))

    def hash_value_short(self, node_ids, seed=1):
        return self.hash_array(node_ids, seed)%(self.memory_dim//4)
    def hash_value_long(self, node_ids, seed=2):
        return self.hash_array(node_ids, seed)%self.memory_dim+self.memory_dim//4

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids).to(torch.long)].long()

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        node_ids = torch.from_numpy(node_ids).unsqueeze(1).repeat(1, updated_node_memories.shape[1]).to(torch.long)

        hash_short = self.hash_value_short(updated_node_memories).astype(np.int)
        # hash_short2 = self.hash_value_short(updated_node_memories,2).astype(np.int)
        hash_long = self.hash_value_long(updated_node_memories).astype(np.int)
        # hash_long2 = self.hash_value_long(updated_node_memories,3).astype(np.int)

        updated_node_memories = torch.from_numpy(updated_node_memories).to(torch.long).to(self.node_memories.device)

        self.node_memories[node_ids, hash_short] = updated_node_memories
        # self.node_memories[node_ids, hash_short2] = updated_node_memories
        self.node_memories[node_ids, hash_long] = updated_node_memories
        # self.node_memories[node_ids, hash_long2] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        return self.node_memories.data.clone()

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data = backup_memory_bank.clone()

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])

class CNEN(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, output_dim: int, update_neighbor: bool = True, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, memory_dim : int = 64, device: str = 'cpu'):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(CNEN, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.memory_dim = memory_dim
        self.device = device
        self.update_neighbor = update_neighbor

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.memory_bank = MemoryBank(self.node_raw_features.shape[0], self.memory_dim)

        self.structure_feat_dim = 5

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'structure': nn.Linear(in_features=self.structure_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        })

        self.num_channels = 4

        self.fusion_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(self.num_channels * self.channel_embedding_dim, self.num_channels * self.channel_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.num_channels * self.channel_embedding_dim))
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=(self.num_channels) * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, positive = False):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # read src and dst memory (batch_size, 1, memory_dim)
        src_memory = self.memory_bank.get_memories(src_node_ids).unsqueeze(1)
        dst_memory = self.memory_bank.get_memories(dst_node_ids).unsqueeze(1)

        src_memory_short, src_memory_long = src_memory[:, :, :self.memory_dim // 4], src_memory[:, :,self.memory_dim // 4:]
        dst_memory_short, dst_memory_long = dst_memory[:, :, :self.memory_dim // 4], dst_memory[:, :,self.memory_dim // 4:]

        # read src and dst neighbor memory (batch_size, max_seq_length, memory_dim)
        src_neighbor_memory = self.memory_bank.get_memories(src_padded_nodes_neighbor_ids)
        dst_neighbor_memory = self.memory_bank.get_memories(dst_padded_nodes_neighbor_ids)

        src_neighbor_memory_short, src_neighbor_memory_long = src_neighbor_memory[:, :,:self.memory_dim // 4], src_neighbor_memory[:, :,self.memory_dim // 4:]
        dst_neighbor_memory_short, dst_neighbor_memory_long = dst_neighbor_memory[:, :,:self.memory_dim // 4], dst_neighbor_memory[:, :,self.memory_dim // 4:]

        # compute co neighbor encoding (batch_size, max_seq_length, 1)
        pos_feature_src_src = torch.sum((src_memory_long.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_long) * (src_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_src_dst = torch.sum((dst_memory_long.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_long) * (src_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)

        pos_feature_dst_dst = torch.sum((dst_memory_long.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_long) * (dst_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_dst_src = torch.sum((src_memory_long.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_long) * (dst_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)

        pos_feature_src_src_short = torch.sum((src_memory_short.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_short) * (src_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_src_dst_short = torch.sum((dst_memory_short.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_short) * (src_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)

        pos_feature_dst_dst_short = torch.sum((dst_memory_short.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_short) * (dst_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_dst_src_short = torch.sum((src_memory_short.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_short) * (dst_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)

        src_coocur = ((torch.from_numpy(dst_node_ids).unsqueeze(1).repeat(1, src_padded_nodes_neighbor_ids.shape[
            1])) == torch.from_numpy(src_padded_nodes_neighbor_ids)).float().to(self.device).unsqueeze(-1)
        dst_coocur = ((torch.from_numpy(src_node_ids).unsqueeze(1).repeat(1, dst_padded_nodes_neighbor_ids.shape[
            1])) == torch.from_numpy(dst_padded_nodes_neighbor_ids)).float().to(self.device).unsqueeze(-1)

        # compute structure feature (batch_size, max_seq_length, 2)
        src_padded_nodes_neighbor_structure_features = torch.cat([pos_feature_src_src, pos_feature_src_dst, pos_feature_src_src_short, pos_feature_src_dst_short, src_coocur], dim=-1)
        dst_padded_nodes_neighbor_structure_features = torch.cat([pos_feature_dst_dst, pos_feature_dst_src, pos_feature_dst_dst_short, pos_feature_dst_src_short, dst_coocur], dim=-1)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_padded_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_padded_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_padded_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_structure_features = self.projection_layer['structure'](src_padded_nodes_neighbor_structure_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_padded_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_padded_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_padded_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_structure_features = self.projection_layer['structure'](dst_padded_nodes_neighbor_structure_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_structure_features = torch.cat([src_patches_nodes_neighbor_structure_features, dst_patches_nodes_neighbor_structure_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
                        patches_nodes_neighbor_time_features, patches_nodes_neighbor_structure_features]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for fusion_layer in self.fusion_layer:
            patches_data = fusion_layer(patches_data)

        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(torch.cat([src_patches_data],dim=-1))
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(torch.cat([dst_patches_data],dim=-1))

        if positive:
            self.memory_bank.set_memories(src_node_ids.flatten(), dst_padded_nodes_neighbor_ids)
            self.memory_bank.set_memories(dst_node_ids.flatten(), src_padded_nodes_neighbor_ids)

            if self.update_neighbor:
                self.memory_bank.set_memories(src_padded_nodes_neighbor_ids[:,0:].flatten(), np.expand_dims(dst_node_ids, 1).repeat(src_padded_nodes_neighbor_ids.shape[1],0))
                self.memory_bank.set_memories(dst_padded_nodes_neighbor_ids[:,0:].flatten(), np.expand_dims(src_node_ids, 1).repeat(dst_padded_nodes_neighbor_ids.shape[1],0))


        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size  == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.long)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.long)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids).long()]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids).long()]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
