# You need to write down your own code here
# Task: Given any head entity name (e.g. Q30) and relation name (e.g. P36), you need to output the top 10 closest tail entity names.
# File entity2vec.vec and relation2vec.vec are 50-dimensional entity and relation embeddings.
# If you use the embeddings learned from Problem 1, you will get extra credits.

import numpy as np
import sklearn.metrics as m
import numpy.linalg as ln


def readfile(name, skip=True):
    entity2id = []
    with open(name) as f:
        if skip:
            f.readline()
        for line in f:
            line = line.split("\n")[0].strip()
            entity2id.append(tuple(line.split("\t")))
    return entity2id


def top_relevant(query, k=3, flag="tail"):

    # similarities = m.pairwise.cosine_similarity(relation2vec, query)
    similarities = []
    if flag == "tail":
        # find an entity
        for e in range(entity2vec.shape[0]):
            var = query - entity2vec[e]
            similarities.append(np.linalg.norm(var, ord=1))
    else:
        # find a relation
        for r in range(relation2vec.shape[0]):
            # print(query.shape, entity2vec[e].shape)
            var = query - relation2vec[r]
            similarities.append(np.linalg.norm(var, ord=1))

    similarities = [(i, j) for i, j in enumerate(similarities)]
    similarities = sorted(similarities, key=lambda u: u[1], reverse=False)
    # print(similarities)
    topk = [str(i[0]) for i in similarities[0:k]]
    # t_index = topk[0]
    # tail = id2entity[t_index]
    if flag == "tail":
        res = [id2entity[i] for i in topk]
    else:
        res = [id2relation[i] for i in topk]
    return res


def f_query_tail(head="Q30", relation="P36", k=10):
    h = entity2vec[int(entity2id[head])]
    r = relation2vec[int(relation2id[relation])]
    query = h + r
    # query = np.reshape(h + r, (1, -1))
    tail = top_relevant(query, k, "tail")
    # t = entity2vec[int(t_index)]
    for t in tail:
        print(head, relation, t)


def f_query_relation(head="Q30", tail="Q49", k=10):
    h = entity2vec[int(entity2id[head])]
    t = entity2vec[int(entity2id[tail])]
    query = t - h
    # query = np.reshape(t - h, (1, -1))

    relation = top_relevant(query, k, "relation")
    # r = relation2vec[int(r_index)]
    for r in relation:
        print(head, r, tail)


if __name__ == "__main__":

    entity2id = readfile("./data/entity2id.txt")
    relation2id = readfile("./data/relation2id.txt")

    entity2id = {i[0]: i[1] for i in entity2id}
    relation2id = {i[0]: i[1] for i in relation2id}

    entity2vec = readfile("./data/entity2vec.vec", False)
    relation2vec = readfile("./data/relation2vec.vec", False)

    entity2vec = np.array([list(i) for i in entity2vec], dtype=np.float32)
    relation2vec = np.array([list(i) for i in relation2vec], dtype=np.float32)

    id2entity = {entity2id[k]: k for k in entity2id}
    id2relation = {relation2id[k]: k for k in relation2id}

    print("result first query: ")
    out = f_query_tail(head="Q30", relation="P36")

    print("result second query: ")
    out = f_query_relation(head="Q30", tail="Q49")
