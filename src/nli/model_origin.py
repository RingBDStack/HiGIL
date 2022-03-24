import math
import logging
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcontrib.optim import SWA
import dgl
from dgl.nn import GraphConv
from dgl.nn import WeightAndSum
from transformers import RobertaModel, XLNetModel
import config
from utils import get_current_datetime
from utils.common import load_json, save_jsonl, load_multiple_jsonl, load_pickle, save_pickle
from utils.fever import claim_is_same
from nli.dataloader import train_batch_generator, dev_batch_generator, test_batch_generator

logger = logging.getLogger(__name__)

def _weight_and_sum(embeddings_tensor):
    feats = embeddings_tensor
    nodes = feats.shape[0]
    src_ids = torch.IntTensor(nodes * nodes)
    dst_ids = torch.IntTensor(nodes * nodes)
    for i in range(0, nodes):
        for j in range(i * nodes, (i + 1) * nodes):
            src_ids[j] = i

    for i in range(0, nodes):
        for j in range(0, nodes):
            dst_ids[i * nodes + j] = j

    src_ids = src_ids.cuda(0)
    dst_ids = dst_ids.cuda(0)
    feats = feats.cuda(0)

    g = dgl.graph((src_ids, dst_ids))
    weight_and_sum = WeightAndSum(feats.shape[1]).cuda(0)
    res = weight_and_sum(g, feats)

    return res[0].detach()


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class NaturalLanguageInferenceModel(nn.Module):

    def _has_nan(self, embeddings):
        return torch.isnan(embeddings).sum().item() != 0

    def __init__(self, embed_size=1024, activation=F.relu, dropout=0.1, class_num=3):
        super(NaturalLanguageInferenceModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained('FactChecking/src/nli/pretrained_model/roberta-large/', local_files_only=True)


        self.all_pconv, self.all_vconv, self.all_sconv = [
            self._stacked_graph_conv(embed_size, activation, 1),
            self._stacked_graph_conv(embed_size, activation, 1),
            self._stacked_graph_conv(embed_size, activation, 1),
        ]

        self.w_all = nn.Linear(embed_size, embed_size)
        self.w_align = nn.Linear(4 * embed_size, embed_size)

        self.cls_classifier = nn.Linear(embed_size, class_num)
        self.classifier = nn.Linear(2 * embed_size, class_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, all_graph):
        input_id = tokens['input_ids']
        attn_mask = tokens['attention_mask']
        embeddings = self.encoder(**tokens)[0]
        assert not self._has_nan(embeddings)

        cls_embed = embeddings[:, 0, :]

        self._initialize_pgraph(embeddings, tokens, all_graph, segment_idx=1)

        all_pembeds = self._graph_conv(all_graph['pgraph'], self.all_pconv)
        p_feat, p_attn = self._graph_cross_attn('p', all_pembeds, all_pembeds, all_graph['pclaim_nodes'], all_graph['pnodes'], all_graph['pgraph'])

        self._initialize_vgraph(all_pembeds, all_graph)

        all_vembeds = self._graph_conv(all_graph['vgraph'], self.all_vconv)
        
        v_feat, v_attn = self._graph_cross_attn('v', all_vembeds, all_vembeds, all_graph['vclaim_nodes'], all_graph['vnodes'], all_graph['vgraph'])

        self._initialize_sgraph(all_pembeds, all_graph)

        all_sembeds = self._graph_conv(all_graph['sgraph'], self.all_sconv)
        s_feat, s_attn = self._graph_cross_attn('s', all_sembeds, all_sembeds, all_graph['sclaim_nodes'], all_graph['snodes'], all_graph['sgraph'])

        return self.cls_classifier(cls_embed), self._classify(cls_embed, p_feat, v_feat, s_feat) #, [p_attn, v_attn, s_attn]

    def _stacked_graph_conv(self, feats, activation, layer_num):
        return nn.ModuleList([GraphConv(feats, feats, activation=activation) for _ in range(layer_num)])

    def _graph_conv(self, graph, conv_layers):
        feat = graph.ndata['feat']
        for idx, conv_layer in enumerate(conv_layers):
            if idx != 0:
                feat = self.dropout(feat)
            feat = conv_layer(graph, feat)
        return feat

    def _graph_toporeach(self, graph, claim_nodes, evidence_nodes):
        edges = graph.edges()
        edges_start = edges[0].detach().cpu().numpy()
        edges_end = edges[1].detach().cpu().numpy()
        degree = [[]for i in range(max(max(edges_start), max(edges_end)) + 1)]
        for i in range(len(edges_end)):
            degree[edges_start[i]].append(edges_end[i])
            degree[edges_end[i]].append(edges_start[i])
        for i in range(len(degree)):
            degree[i] = list(set(degree[i]))
            degree[i].sort()

        step = int(len(evidence_nodes) / 4 + 2)
        p = [[[0 for i in range(evidence_nodes[-1] + 1)] for j in range(claim_nodes[-1] + 1)] for t in range(step)]
        first_claim = claim_nodes[0]
        first_evidence = evidence_nodes[0]

        for claim in claim_nodes:
            for evidence in degree[claim]:
                p[1][claim][evidence] = 1 / len(degree[claim])

        for t in range(2, step):
            for claim in claim_nodes:
                for evidence in evidence_nodes:
                    for evidence_t in degree[evidence]:
                        p[t][claim][evidence] += p[t - 1][claim][evidence_t] / len(degree[evidence])

        tmp_p = [[[0 for i in range(len(evidence_nodes))]for j in range(len(claim_nodes))] for t in range(step)]

        for t in range(step):
            for i in range(len(claim_nodes)):
                for j in range(len(evidence_nodes)):
                    tmp_p[t][i][j] = p[t][i + first_claim][j + first_evidence]

        res = torch.tensor(tmp_p[step - 1])
        res = F.softmax(res)
        return res


    def _graph_cross_attn(self, sign, claim_embeds, evidence_embeds, claim_nodes, evidence_nodes, graph):
        batch_aligned = []
        for claim_node, evidence_node in zip(claim_nodes, evidence_nodes):
            claim_embed = claim_embeds[claim_node]
            evidence_embed = evidence_embeds[evidence_node]

            scores = torch.matmul(
                self.w_all(claim_embed),
                self.w_all(evidence_embed).transpose(-2, -1)
            )
            attn = F.softmax(scores, dim=-1)
            #attn = self._graph_toporeach(graph, claim_node, evidence_node).cuda(0)
            p_attn = self.dropout(attn)
            claim_centric = torch.matmul(p_attn, evidence_embed)
            aligned = self.w_align(torch.cat([
                claim_embed, claim_centric, claim_embed - claim_centric, claim_embed * claim_centric
            ], axis=-1))
            batch_aligned.append(aligned.mean(axis=0))
        return torch.stack(batch_aligned), attn

    def _classify(self, emb_cls, p_feat, v_feat, s_feat):
        return [self.classifier(torch.cat([emb_cls, feat], axis=-1)) for feat in [p_feat, v_feat, s_feat]]

    

    def _initialize_pgraph(self, embeddings, tokens, graph, segment_idx):
        pgraph, batch_p2words = graph['pgraph'], graph['p2words']

        node_embeds = []
        for batch_idx, p2words in enumerate(batch_p2words):
            # Warning: The token sequence of a node is not necessarily contiguous!
            for p, words in p2words.items():
                start, end = math.inf, 0
                for word_idx in words:
                    tokenspan = tokens.word_to_tokens(batch_idx, word_idx, segment_idx)
                    if tokenspan is None:
                        # Warning: Sequence length exceeds 512, node truncated!
                        print('This should not be happening for XLNet!')
                        start, end = 511, 512
                        break

                    start = min(start, tokenspan.start)
                    end = max(end, tokenspan.end)
                #node_embeds.append(embeddings[batch_idx, start:end, :].mean(axis=0))
                #node_embeds.append(embeddings[batch_idx, start:end, :].sum(axis=0))
                #node_embeds.append(embeddings[batch_idx, start:end, :].max(axis=0)[0])
                node_embeds.append(_weight_and_sum(embeddings[batch_idx, start:end, :]))
        pgraph.ndata['feat'] = torch.stack(node_embeds)

    def _initialize_vgraph(self, embeddings, graph):
        vgraph, v2p = graph['vgraph'], graph['v2p']

        node_embeds = []
        for v, pnodes in v2p.items():
            assert len(pnodes) != 0
            #node_embeds.append(embeddings[pnodes].mean(axis=0))
            #node_embeds.append(embeddings[pnodes].sum(axis=0))
            #node_embeds.append(embeddings[pnodes].max(axis=0)[0])
            node_embeds.append(_weight_and_sum(embeddings[pnodes]))
        vgraph.ndata['feat'] = torch.stack(node_embeds)

    def _initialize_sgraph(self, embeddings, graph):
        sgraph, s2v = graph['sgraph'], graph['s2v']

        node_embeds = []
        for s, vnodes in s2v.items():
            assert len(vnodes) != 0
            #node_embeds.append(embeddings[vnodes].mean(axis=0))
            #node_embeds.append(embeddings[vnodes].sum(axis=0))
            #node_embeds.append(embeddings[vnodes].max(axis=0)[0])
            node_embeds.append(_weight_and_sum(embeddings[vnodes]))
        sgraph.ndata['feat'] = torch.stack(node_embeds)


class NaturalLanguageInferenceExperiment:

    def __init__(self, model_dir=config.NLI_MODEL_DIR):
        self.model = NaturalLanguageInferenceModel()
        self.model_dir = model_dir
        self.fgm = FGM(self.model)

    def from_checkpoint(self, ckpt_name, device=config.DEVICE):
        ckpt_path = Path(self.model_dir) / (ckpt_name + '-1.pt')

        state_dict = torch.load(ckpt_path, device)
        if ckpt_name == 'pvs':
            model = NaturalLanguageInferenceModel()
            model.to(device)
            state_dict['cls_classifier.weight'] = model.cls_classifier.weight
            state_dict['cls_classifier.bias'] = model.cls_classifier.bias
        self.model.load_state_dict(state_dict)
     
    def set_parameters(self, params):
        self.bertparams = []
        self.graphparams = []

        for k,p in params:
            if p.requires_grad:
                if 'encoder' in k:
                    self.bertparams.append(p)
                else:
                    self.graphparams.append(p)
        self.params = [
            {'params':self.bertparams, 'lr': config.NLI_TRAIN_LR_BERT},
            {'params':self.graphparams, 'lr': config.NLI_TRAIN_LR_GRAPH}
        ]
        return self.params

    def backward_step(self, batch, optimizer, labels, step):
        out_cls, (out_p, out_v, out_s) = self.model(**batch)
        # self.model(**batch)
        # continue

        if ckpt_name == 'cls':
            loss = F.cross_entropy(out_cls, labels)
            out = out_cls
        elif ckpt_name == 'p':
            loss = F.cross_entropy(out_p, labels)
            out = out_p
        elif ckpt_name == 'pv':
            loss_p, loss_v = [F.cross_entropy(out, labels) for out in [out_p, out_v]]
            loss = (loss_p + loss_v) / 2
            out = (out_p + out_v) / 2
        elif ckpt_name == 'ps':
            loss_p, loss_s = [F.cross_entropy(out, labels) for out in [out_p, out_s]]
            loss = (loss_p + loss_s) / 2
            out = (out_p + out_s) / 2
        else:
            loss_p, loss_v, loss_s = [F.cross_entropy(out, labels) for out in [out_p, out_v, out_s]]
            loss = (loss_p + loss_v + loss_s) / 3
            out = (out_p + out_v + out_s) / 3

        logger.info(f'Batch loss at step {step}: {loss.item()}.')

        preds = out.argmax(axis=-1)
        loss.backward()
        return out, preds

    def train(self,
            ckpt_name,
            train_batch_generator=train_batch_generator,
            epochs=config.NLI_TRAIN_EPOCHS,
            device=config.DEVICE):
        self.model.to(device)
        params = self.set_parameters(self.model.named_parameters())
        optimizer = torch.optim.Adam(params, lr=config.NLI_TRAIN_LR_BERT)
        optimizer = SWA(optimizer, swa_start=20, swa_freq=10, swa_lr=config.NLI_TRAIN_LR_BERT)
        
        for epoch in range(epochs):
            self.model.train()
            correct_cnt, total_cnt = 0, 0
            for step, batch in enumerate(train_batch_generator()):
                labels = batch.pop('label')
                optimizer.zero_grad()
                out, preds = self.backward_step(batch, optimizer, labels, step)
                correct_cnt += (preds == labels).sum()
                total_cnt += out.size(0)

                self.fgm.attack()
                out, preds = self.backward_step(batch, optimizer, labels, step)
                correct_cnt += (preds == labels).sum()
                total_cnt += out.size(0)
                self.fgm.restore()
                optimizer.step()

            optimizer.swap_swa_sgd()
            label_acc = correct_cnt / total_cnt
            logger.info(f'Label accuracy at epoch: {epoch}: {label_acc}.')

            self.evaluate(ckpt_name, save=False)

            ckpt_path = Path(self.model_dir) / f'{ckpt_name}-{epoch}.pt'
            torch.save(self.model.state_dict(), ckpt_path)
            logger.info(f'Checkpoint saved to {ckpt_path}.')
        

    def evaluate(self,
                 ckpt_name,
                 dev_batch_generator=dev_batch_generator,
                 device=config.DEVICE,
                 dev_nli_path=config.DEV_NLI_PATH,
                 save=True):
        self.model.to(device)
        self.model.eval()

        correct_cnt, total_cnt = 0, 0
        nli_result = []
        with torch.no_grad():
            for batch in dev_batch_generator():
                claims = batch.pop('claim')
                labels = batch.pop('label')

                out_cls, (out_p, out_v, out_s) = self.model(**batch)


                if ckpt_name == 'cls':
                    out = out_cls
                elif ckpt_name == 'p':
                    out = out_p
                elif ckpt_name == 'pv':
                    out = (out_p + out_v) / 2
                elif ckpt_name == 'ps':
                    out = (out_p + out_s) / 2
                else:
                    out = (out_p + out_v + out_s) / 3
                preds = out.argmax(axis=-1)

                correct_cnt += (preds == labels).sum()
                total_cnt += out.size(0)

                nli_result.extend(
                    self._pack_dev_nli_result(claims, labels.tolist(), preds.tolist())
                )


        fever_acc = correct_cnt / total_cnt
        logger.info(f'FEVER label accuracy: {fever_acc}.')

        if save:
            save_jsonl(nli_result, dev_nli_path)
            logger.info(f'NLI result saved to {dev_nli_path}.')

    def test(self,
             ckpt_name,
             test_batch_generator=test_batch_generator,
             device=config.DEVICE,
             test_nli_path=config.TEST_NLI_PATH):
        self.model.to(device)
        self.model.eval()

        nli_result = []
        with torch.no_grad():
            for batch in test_batch_generator():
                claims = batch.pop('claim')

                out_cls, (out_p, out_v, out_s) = self.model(**batch)

                if ckpt_name == 'cls':
                    out = out_cls
                elif ckpt_name == 'p':
                    out = out_p
                elif ckpt_name == 'pv':
                    out = (out_p + out_v) / 2
                elif ckpt_name == 'ps':
                    out = (out_p + out_s) / 2
                else:
                    out = (out_p + out_v + out_s) / 3
                preds = out.argmax(axis=-1)

                nli_result.extend(
                    self._pack_test_nli_result(claims, preds.tolist())
                )
        save_jsonl(nli_result, test_nli_path)
        logger.info(f'NLI result saved to: {test_nli_path}.')

    def merge_test(nli_path, sent_sel_path, fever_path):
        nli_dps = load_multiple_jsonl(nli_path)
        sent_sel_dps = load_multiple_jsonl(sent_sel_path)
        fever_dps = load_multiple_jsonl(fever_path)
        assert len(nli_dps) == len(sent_sel_dps) == len(fever_dps)

        for nli_dp, sent_sel_dp, fever_dp in tqdm(zip(nli_dps, sent_sel_dps, fever_dps), total=len(nli_dps)):
            assert claim_is_same(nli_dp['claim'], sent_sel_dp['claim'])
            assert claim_is_same(nli_dp['claim'], fever_dp['claim'])

            nli_dp.pop('claim')
            nli_dp['id'] = fever_dp['id']
            nli_dp['predicted_label'] = nli_dp.pop('predicated_label')
            nli_dp['predicted_evidence'] = [pred for idx, pred in enumerate(sent_sel_dp['predicted_evidence'])]
        save_jsonl(nli_dps, 'predictions.jsonl')

    @staticmethod
    def _pack_dev_nli_result(claims, labels, preds, label_map=load_json(config.FEVER_LABEL_PATH)):
        assert len(claims) == len(labels) == len(preds)
        reverse_map = {v: k for k, v in label_map.items()}

        results = []
        for claim, label, pred in zip(claims, labels, preds):
            results.append({
                'claim': claim,
                'label': reverse_map[label],
                'predicated_label': reverse_map[pred]
            })
        return results

    @staticmethod
    def _pack_test_nli_result(claims, preds, label_map=load_json(config.FEVER_LABEL_PATH)):
        assert len(claims) == len(preds)
        reverse_map = {v: k for k, v in label_map.items()}

        results = []
        for claim, pred in zip(claims, preds):
            results.append({
                'claim': claim,
                'predicated_label': reverse_map[pred]
            })
        return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_name', choices=['cls', 'p', 'pv', 'ps', 'pvs', 'xlnet'], required=True)
    parser.add_argument('--model', choices=['train', 'from_ckpt'], required=True)
    parser.add_argument('--evaluate', choices=['train', 'dev', 'test'], required=True)
    args = parser.parse_args()
    ckpt_name, model, evaluate = args.ckpt_name, args.model, args.evaluate

    log_path = config.NLI_LOGGING_DIR / f'{Path(__file__).stem}-{get_current_datetime()}'
    logging.basicConfig(filename=log_path, level=logging.INFO)

    ckpt_train_path = \
        config.NLI_DIR / ckpt_name / config.TRAIN_NLI_PATH.name
    ckpt_dev_path = \
        config.NLI_DIR / ckpt_name / config.DEV_NLI_PATH.name
    ckpt_test_path = \
        config.NLI_DIR / ckpt_name / config.TEST_NLI_PATH.name

    if not ckpt_train_path.parent.exists():
        ckpt_train_path.parent.mkdir()

    nlie = NaturalLanguageInferenceExperiment()
    if model == 'train':
        nlie.train(ckpt_name)
    else:
        nlie.from_checkpoint(ckpt_name)

    if evaluate == 'dev':
        nlie.evaluate(ckpt_name, dev_nli_path=ckpt_dev_path)
    else:
        nlie.test(ckpt_name, test_nli_path=ckpt_test_path)
        nlie.merge_test(ckpt_test_path, config.TEST_SENT_SEL_PATH, config.FEVER_TEST_PATH)
