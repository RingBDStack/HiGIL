import random
from tqdm import tqdm
import torch
import dgl
from transformers import RobertaTokenizerFast, XLNetTokenizerFast
import config
import math
from utils.common import load_json, load_pickle


tokenizer = RobertaTokenizerFast.from_pretrained('FactChecking/src/nli/pretrained_model/roberta-large/', add_prefix_space=True,model_max_length=512)


def train_batch_generator(nli_graph_path=config.NLI_TRAIN_GRAPH_PATH,
                          batch_size=config.NLI_TRAIN_BATCH_SIZE,
                          device=config.DEVICE,
                          label_map=load_json(config.FEVER_LABEL_PATH)):
    dps = load_pickle(nli_graph_path)
    random.shuffle(dps)
    for start in tqdm(range(0,len(dps),batch_size), desc='Generating Training Batch'):
            end = start + batch_size
            batch_words = [dp['words'] for dp in dps[start:end]]
            batch_claim, batch_evidence = list(zip(*batch_words))
            batch_tokens = tokenizer(
                batch_claim, batch_evidence,
                padding=True, truncation=True, is_split_into_words=True, return_tensors='pt'
            )

            all_graph = _batch_graph([dp['evidence_graph'] for dp in dps[start:end]], device)

            batch_label = [label_map[dp['label']] for dp in dps[start:end]]

            yield {
                'tokens': batch_tokens.to(device),
                'all_graph': all_graph,
                'label': torch.tensor(batch_label, device=device)
            }


def dev_batch_generator(nli_graph_path=config.NLI_DEV_GRAPH_PATH,
                        batch_size=config.NLI_INFERENCE_BATCH_SIZE,
                        device=config.DEVICE,
                        label_map=load_json(config.FEVER_LABEL_PATH)):
    '''
    Differences between train_batch_generator:
        - Dataset is shuffled;
        - A batch contains textual claim. (To re-construct predicted result.)
    '''

    dps = load_pickle(nli_graph_path)
    for start in tqdm(range(0, len(dps), batch_size), desc='Generating Development Batch'):
        end = start + batch_size

        batch_words = [dp['words'] for dp in dps[start:end]]
        batch_claim, batch_evidence = list(zip(*batch_words))
        batch_tokens = tokenizer(
            batch_claim, batch_evidence, 
            padding=True, truncation=True, is_split_into_words=True, return_tensors='pt'
        )


        all_graph = _batch_graph([dp['evidence_graph'] for dp in dps[start:end]], device)

        batch_label = [label_map[dp['label']] for dp in dps[start:end]]

        # Textual claim here.
        batch_claim = [dp['claim'] for dp in dps[start:end]]

        yield {
            'tokens': batch_tokens.to(device),
            'all_graph': all_graph,
            'label': torch.tensor(batch_label, device=device),
            'claim': batch_claim,
        }


def test_batch_generator(nli_graph_path=config.NLI_TEST_GRAPH_PATH,
                         batch_size=config.NLI_INFERENCE_BATCH_SIZE,
                         device=config.DEVICE):
    '''
    Difference between dev_batch_generator:
        - A batch does not contain label.
    '''

    dps = load_pickle(nli_graph_path)
    for start in tqdm(range(0, len(dps), batch_size), desc='Generating Test Batch'):
        end = start + batch_size

        batch_words = [dp['words'] for dp in dps[start:end]]
        batch_claim, batch_evidence = list(zip(*batch_words))
        batch_tokens = tokenizer(
            batch_claim, batch_evidence, 
            padding=True, truncation=True, is_split_into_words=True, return_tensors='pt'
        )

        all_graph = _batch_graph([dp['evidence_graph'] for dp in dps[start:end]], device)

        # Textual claim here.
        batch_claim = [dp['claim'] for dp in dps[start:end]]

        yield {
            'tokens': batch_tokens.to(device),
            'all_graph': all_graph,
            'claim': batch_claim
        }


def _batch_graph(batch_graph, device):
    '''
    Merge all graphs into a large graph, node IDs re-ordered.
    '''

    # batch_p2words is list for it maps to sentences (not graphs).
    batch_p2words, batch_v2p, batch_s2v = [], {}, {}
    batch_pnodes, batch_vnodes, batch_snodes = [], [], []
    batch_pclaim_nodes, batch_vclaim_nodes, batch_sclaim_nodes = [], [], []
    poffset, voffset, soffset = 0, 0, 0
    for batch_idx, graph in enumerate(batch_graph):
        (p2words, pedges), (v2p, vedges), (s2v, sedges) = \
            graph['phrase_graph'], graph['verb_graph'], graph['sentence_graph']

        # Update graph.
        if batch_idx == 0:
            batch_pgraph = dgl.graph(pedges, num_nodes=len(p2words))
            batch_vgraph = dgl.graph(vedges, num_nodes=len(v2p))
            batch_sgraph = dgl.graph(sedges, num_nodes=len(s2v))
        else:
            batch_pgraph = dgl.batch([batch_pgraph, dgl.graph(pedges, num_nodes=len(p2words))])
            batch_vgraph = dgl.batch([batch_vgraph, dgl.graph(vedges, num_nodes=len(v2p))])
            batch_sgraph = dgl.batch([batch_sgraph, dgl.graph(sedges, num_nodes=len(s2v))])

        # Update mappings across different levels of graphs. (For pooling.)
        batch_p2words.append(_offset_keys(p2words, poffset))
        _update_mapping(batch_v2p, v2p, voffset, poffset)
        _update_mapping(batch_s2v, s2v, soffset, voffset)

        # Record graph node IDs. (For corss-attention.)
        batch_pnodes.append([poffset + p for p in p2words.keys()])
        batch_vnodes.append([voffset + v for v in v2p.keys()])
        batch_snodes.append([soffset + s for s in s2v.keys()])

        batch_pclaim_nodes.append([poffset + p for p in graph['pclaim_node']])
        batch_vclaim_nodes.append([voffset + v for v in graph['vclaim_node']])
        batch_sclaim_nodes.append([soffset + s for s in graph['sclaim_node']])

        poffset += len(p2words)
        voffset += len(v2p)
        soffset += len(s2v)

    # Add self-loop. (GraphConv forbids zero-degree nodes.)
    batch_pgraph = batch_pgraph.add_self_loop()
    batch_vgraph = batch_vgraph.add_self_loop()
    batch_sgraph = batch_sgraph.add_self_loop()

    return {
        'pgraph': batch_pgraph.to(device),
        'vgraph': batch_vgraph.to(device),
        'sgraph': batch_sgraph.to(device),
        'p2words': batch_p2words,
        'v2p': batch_v2p,
        's2v': batch_s2v,
        'pnodes': batch_pnodes,
        'vnodes': batch_vnodes,
        'snodes': batch_snodes,
        'pclaim_nodes': batch_pclaim_nodes,
        'vclaim_nodes': batch_vclaim_nodes,
        'sclaim_nodes': batch_sclaim_nodes,
    }


def _offset_keys(k2v, koffset):
    return {koffset + k: v for k, v in k2v.items()}


def _update_mapping(batch_k2v, k2v, koffset, voffset):
    for k, vs in k2v.items():
        assert koffset + k not in batch_k2v.keys()
        batch_k2v[koffset + k] = [voffset + v for v in vs]

