import torch
import torch.nn as nn
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np


def expact_test(RC):
    sum_index = []
    sum_col = []
    for i in range(0, RC.shape[0]):
        sum_index.append(sum(RC[i, :]))
    for j in range(0, RC.shape[1]):
        sum_col.append(sum(RC[:, j]))
    sum_total = sum(sum_col)
    expact = np.zeros(RC.shape)
    for i in range(0, RC.shape[0]):
        for j in range(0, RC.shape[1]):
            expact[i][j] = sum_index[i] * sum_col[j] / sum_total
    return np.any(expact < 5)


def relevance_test(stu_skill, p=0.05):
    Cov = np.zeros([stu_skill.shape[1], stu_skill.shape[1]])
    for i in range(stu_skill.shape[1]):
        for j in range(stu_skill.shape[1]):
            if i == j:
                Cov[i][j] = 1
                continue
            RC = np.array([[0.0, 0.0], [0.0, 0.0]])
            for stu in range(stu_skill.shape[0]):
                if stu_skill[stu][i] == 1:
                    if stu_skill[stu][j] == 1:
                        RC[0][0] += 1
                    elif stu_skill[stu][j] == -1:
                        RC[0][1] += 1
                elif stu_skill[stu][i] == -1:
                    if stu_skill[stu][j] == 1:
                        RC[1][0] += 1
                    elif stu_skill[stu][j] == -1:
                        RC[1][1] += 1
            if not (np.all(RC)):
                continue
            if expact_test(RC):
                if p >= fisher_exact(RC)[1]:
                    Cov[i][j] = 1
            else:
                if p >= chi2_contingency(RC)[1]:
                    Cov[i][j] = 1
    return Cov


class ASRCD(nn.Module):
    def __init__(self,
                 student_n,
                 skill_k,
                 exer_e,
                 embed_dim,
                 cov,
                 hidden_dim_1,
                 hidden_dim_2,
                 hidden_dim_3=64,
                 hidden_dim_4=32,
                 res_mode=True,
                 is_deep=True,
                 relation_type='None',
                 dropout=0.3,
                 bia=True,
                 device=None):
        super(ASRCD, self).__init__()
        self.device = device
        self.K = skill_k
        self.N = student_n
        self.E = exer_e
        self.D = embed_dim
        self.lamb_layer = nn.Linear(2 * self.K, 1, bias=bia)
        self.is_res = res_mode
        self.is_deep = is_deep

        self.dense1 = nn.Sequential(nn.Linear(self.K, hidden_dim_1, bias=bia),
                                    nn.Sigmoid())
        self.drop_1 = nn.Dropout(p=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim_1)

        self.dense2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2, bias=bia), nn.Sigmoid())
        self.drop_2 = nn.Dropout(p=dropout)
        self.ln2 = nn.BatchNorm1d(hidden_dim_2)

        if self.is_deep:
            self.dense3 = nn.Sequential(
                nn.Linear(hidden_dim_2, hidden_dim_3, bias=bia), nn.Sigmoid())
            self.drop_3 = nn.Dropout(p=dropout)
            self.ln3 = nn.BatchNorm1d(hidden_dim_3)

            self.dense4 = nn.Sequential(
                nn.Linear(hidden_dim_3, hidden_dim_4, bias=bia), nn.Sigmoid())
            self.drop_4 = nn.Dropout(p=dropout)
            self.ln4 = nn.BatchNorm1d(hidden_dim_4)

            self.dense5 = nn.Sequential(nn.Linear(hidden_dim_4, 1, bias=bia))
        else:
            self.dense3 = nn.Sequential(nn.Linear(hidden_dim_2, 1, bias=bia))

        if self.is_res:
            self.res_linear1 = nn.Linear(self.K, hidden_dim_1)
            self.res_linear2 = nn.Linear(hidden_dim_1, hidden_dim_2)
            if self.is_deep:
                self.res_linear3 = nn.Linear(hidden_dim_2, hidden_dim_3)
                self.res_linear4 = nn.Linear(hidden_dim_3, hidden_dim_4)

        self.relation_type = relation_type
        self.relation_weight = nn.Parameter(nn.randn(self.K, self.K) * cov,
                                            requires_grad=True).to(self.device)
        self.cov = cov

        # stu-related
        self.stu_pro_mat = nn.Parameter(torch.randn(self.K, self.D, self.N),
                                        requires_grad=True).to(self.device)
        # self.concept_embed_pro = nn.Sequential(
        #     [nn.Linear(self.D * 2, self.D, True)])
        self.stu_embedding = nn.Embedding(self.N + 1, self.D).to(self.device)

        # exer-related
        self.concept_embed_exercise = nn.Sequential(
            [nn.Linear(self.D * 2, self.D, True)])
        self.exer_embedding = nn.Embedding(self.E + 1, self.D).to(self.device)
        self.disc_embedding = nn.Embedding(self.E + 1, 1).to(self.device)
        self.disc = nn.Sequential(nn.Linear(self.D, 1))
        self.exer_concept_integration = ConceptIntegration(
            2 * self.D, self.D, self.device)

        # concept-related
        self.concept_neighbor_weight = nn.Sequential(
            [nn.Linear(self.D * 2, 1, True)])
        self.concept_embedding = nn.Parameter(torch.randn(self.K, self.D),
                                              requires_grad=True).to(
                                                  self.device)

        self.stu_integration = NeighborIntegration(self.D, self.cov, self.device)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, stu_id, exer_id, origin_ids):
        stu_id = stu_id.to(self.device)
        origin_ids = origin_ids.to(self.device)
        exer_id = exer_id.to(self.device)

        stu_emb = self.stu_embedding(stu_id)
        exer_emb = self.exer_embedding(exer_id)

        origin_ids = origin_ids @ origin_ids.reshape(
            origin_ids.shape[0], origin_ids.shape[2], origin_ids.shape[1])
        origin_emb = origin_ids @ self.concept_embedding

        neighbors_ids = torch.matmul(origin_ids, self.cov).unsqueeze(-1)
        neighbors_ids = neighbors_ids @ neighbors_ids.reshape(
            neighbors_ids.shape[0], neighbors_ids.shape[1],
            neighbors_ids.shape[3], neighbors_ids.shape[2])
        neighbors_emb = torch.matmul(neighbors_ids, self.concept_embedding)

        disc = self.disc_embedding(exer_id)

        stu_inetgrated = self.stu_integration(stu_emb, origin_emb, neighbors_emb)
        exercise_integrated = self.exer_concept_integration(
            origin_emb, exer_emb)

        self.interaction = torch.sigmoid(disc * (stu_inetgrated - exercise_integrated))
        # input = squeeze(input)
        x = self.dense1(input.float())
        if self.is_res:
            x = x + self.res_linear1(input.float())
        x = self.ln1(x)
        x = self.drop_1(x)
        dense1_x = x

        x = self.dense2(x)
        if self.is_res:
            x = x + self.res_linear2(dense1_x)
        x = self.ln2(x)
        x = self.drop_2(x)
        dense2_x = x

        x = self.dense3(x)

        if self.is_deep:
            if self.is_res:
                x = x + self.res_linear3(dense2_x)
            x = self.ln3(x)
            x = self.drop_3(x)
            dense3_x = x

            x = self.dense4(x)
            if self.is_res:
                x = x + self.res_linear4(dense3_x)
            x = self.ln4(x)
            x = self.drop_4(x)

            x = self.dense5(x)

        x = torch.sigmoid(x)
        return x


class ConceptIntegration(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(ConceptIntegration, self).__init__()
        self.layer1 = nn.Sequential([nn.Linear(input_dim, output_dim)])
        self.output_dim = output_dim
        self.device = device
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, concept, target) -> torch.Tensor:
        index = torch.sum(concept, -1, True)
        index[index != 0] = 1
        index = index.to(self.device)

        concept = concept.to(self.device)
        target = target.to(self.device)

        x = torch.cat([concept, target], 1)
        x = self.layer1(x)
        x = x * index
        return x


class NeighborIntegration(nn.Module):
    def __init__(self, embed_dim, cov, device):
        super(NeighborIntegration, self).__init__()
        self.cov = cov
        self.D = embed_dim
        self.stu_concept_integration = ConceptIntegration(
            2 * self.D, self.D, self.device)
        self.concept_neighbor_weight = ConceptIntegration(
            2 * self.D, 1, self.device)
        self.device = device

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, stu_emb: torch.Tensor, concept_emb: torch.Tensor,
                neighbor_emb: torch.Tensor):
        concept_emb_extend = concept_emb.unsqueeze(-1).repeat(
            1, 1, neighbor_emb.shape[-2], 1)
        stu_emb_extend = stu_emb.unsqueeze(-1).repeat(
            1, 1, neighbor_emb.shape[-2], 1)

        weights = self.concept_neighbor_weight(concept_emb_extend,
                                               neighbor_emb)
        weights_softmax = torch.softmax(weights, -2)
        # weights = weights.reshape(weights.shape[0], weights.shape[1],
        #                           weights.shape[3], weights.shape[2])

        stu_integrated_neighbor = self.stu_concept_integration(stu_emb_extend, neighbor_emb)
        stu_integrated_origin = self.stu_concept_integration(stu_emb_extend, concept_emb_extend)


        y = torch.sum([stu_integrated_origin, weights_softmax * stu_integrated_neighbor], -2)

        return y / 2