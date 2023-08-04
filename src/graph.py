import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import pickle

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


def chi_relevance_test(stu_skill, p=0.05):
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


def build_chi_graph(p):
    with open(r'/home/fishu/IGCD/temp/stu_skill.npy', mode='rb') as f:
        stu_skill = np.load(f)
    cov = chi_relevance_test(stu_skill, p=p)
    with open(r'/home/fishu/IGCD/temp/chi.npy', mode='wb') as f:
        np.save(f, cov)
    return cov


def build_dense_graph(concept_num):
    graph = 1. / (concept_num - 1) * np.ones((concept_num, concept_num))
    np.fill_diagonal(graph, 0)

    with open(r'/home/fishu/IGCD/temp/dense.npy', mode='wb') as f:
        np.save(f, graph)
    return graph


def build_transition_matrix(concept_num):
    with open(r'/home/fishu/IGCD/temp/question_list.pkl', mode='rb') as f:
        question_list = pickle.load(f)
    with open(r'/home/fishu/IGCD/temp/answer_list.pkl', mode='rb') as f:
        answer_list = pickle.load(f)
    with open(r'/home/fishu/IGCD/temp/seq_len_list.pkl', mode='rb') as f:
        seq_len_list = pickle.load(f)

    graph = np.zeros((concept_num, concept_num), dtype=np.int16)
    for i in range(0, len(question_list)):
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            if answer_list[i][j] and answer_list[i][j + 1]:
                continue
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1. / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # T = (graph - np.min(graph)) / (np.max(graph) - np.min(graph))
    # graph = np.array(np.where(T > np.average(graph) * 10 , 1, 0))
    with open(r'/home/fishu/IGCD/temp/transition.npy', mode='wb') as f:
        np.save(f, graph)
    return graph