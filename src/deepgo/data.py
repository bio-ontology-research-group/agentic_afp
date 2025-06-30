import pandas as pd
import torch as th
import numpy as np
import dgl


def get_data(df, terms_dict, features_length):
    """
    Converts dataframe file with protein information and returns
    PyTorch tensors
    """
    data = th.zeros((len(df), features_length), dtype=th.float32)
    labels = {
        "mf": th.zeros((len(df), len(terms_dict["mf"])), dtype=th.float32),
        "bp": th.zeros((len(df), len(terms_dict["bp"])), dtype=th.float32),
        "cc": th.zeros((len(df), len(terms_dict["cc"])), dtype=th.float32),
    }
    for i, row in enumerate(df.itertuples()):
        # Data vector
        data[i, :] = th.FloatTensor(row.esm2_data)
        # Labels vectors
        for go_id in row.prop_annotations:
            if go_id in terms_dict["mf"]:
                g_id = terms_dict["mf"][go_id]
                labels["mf"][i, g_id] = 1
            elif go_id in terms_dict["bp"]:
                g_id = terms_dict["bp"][go_id]
                labels["bp"][i, g_id] = 1
            elif go_id in terms_dict["cc"]:
                g_id = terms_dict["cc"][go_id]
                labels["cc"][i, g_id] = 1
    return data, labels


def load_data(data_root, features_length=2560, test_data_file="test_data.pkl"):
    df = pd.read_pickle(f"{data_root}/mf_terms.pkl")
    terms_mf = df["terms"].values.flatten()
    df = pd.read_pickle(f"{data_root}/bp_terms.pkl")
    terms_bp = df["terms"].values.flatten()
    df = pd.read_pickle(f"{data_root}/cc_terms.pkl")
    terms_cc = df["terms"].values.flatten()
    terms_dict = {
        "mf": {v: i for i, v in enumerate(terms_mf)},
        "bp": {v: i for i, v in enumerate(terms_bp)},
        "cc": {v: i for i, v in enumerate(terms_cc)},
    }

    train_df = pd.read_pickle(f"{data_root}/train_data.pkl")
    valid_df = pd.read_pickle(f"{data_root}/valid_data.pkl")
    test_df = pd.read_pickle(f"{data_root}/{test_data_file}")

    genomes = {}
    for i, row in enumerate(train_df.itertuples()):
        if row.orgs not in genomes:
            genomes[row.orgs] = []
        genomes[row.orgs].append(i)
    for i in genomes:
        genomes[i] = th.tensor(genomes[i], dtype=th.long)

    train_data = get_data(train_df, terms_dict, features_length)
    valid_data = get_data(valid_df, terms_dict, features_length)
    test_data = get_data(test_df, terms_dict, features_length)

    return (
        terms_dict,
        train_data,
        valid_data,
        test_data,
        test_df,
        list(genomes.values()),
    )


def load_ppi_data(
    data_root,
    ont,
    features_length=2560,
    features_column="esm2",
    test_data_file="test_data.pkl",
    ppi_graph_file="ppi_test.bin",
):

    terms_df = pd.read_pickle(f"{data_root}/{ont}/terms.pkl")
    terms = terms_df["gos"].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print("Terms", len(terms))

    mf_df = pd.read_pickle(f"{data_root}/mf/terms.pkl")
    mfs = mf_df["gos"].values
    mfs_dict = {v: k for k, v in enumerate(mfs)}

    ipr_df = pd.read_pickle(f"{data_root}/{ont}/interpros.pkl")
    iprs = ipr_df["interpros"].values
    iprs_dict = {v: k for k, v in enumerate(iprs)}

    feat_dict = None

    if features_column == "interpros":
        features_length = len(iprs_dict)
        feat_dict = iprs_dict
    elif features_column != "esm2":
        features_length = len(mfs_dict)
        feat_dict = mfs_dict

    train_df = pd.read_pickle(f"{data_root}/{ont}/train_data.pkl")
    valid_df = pd.read_pickle(f"{data_root}/{ont}/valid_data.pkl")
    test_df = pd.read_pickle(f"{data_root}/{ont}/{test_data_file}")

    df = pd.concat([train_df, valid_df, test_df])
    graphs, nids = dgl.load_graphs(f"{data_root}/{ont}/{ppi_graph_file}")

    data, labels = get_data(df, feat_dict, terms_dict, features_length, features_column)
    graph = graphs[0]
    graph.ndata["feat"] = data
    graph.ndata["labels"] = labels
    train_nids, valid_nids, test_nids = (
        nids["train_nids"],
        nids["valid_nids"],
        nids["test_nids"],
    )
    return (
        feat_dict,
        terms_dict,
        graph,
        train_nids,
        valid_nids,
        test_nids,
        data,
        labels,
        test_df,
    )


def load_normal_forms(go_file):
    """
    Parses and loads normalized (using Normalize.groovy script)
    ontology axioms file
    Args:
        go_file (string): Path to a file with normal forms
    Returns:
        nf1 (list): List of tuples with 2 elements
        nf2 (list): List of tuples with 3 elements
        nf3 (list): List of tuples with 3 elements
        nf4 (list): List of tuples with 3 elements
        relations (dict): Dictionary with relation names
        classes (dict): Dictionary with class names
    """
    nf1 = []
    nf2 = []
    nf2_bot = []
    nf3 = []
    nf4 = []
    relations = {}
    classes = {}

    def get_index(go_id):
        if go_id not in classes:
            classes[go_id] = len(classes)
        return classes[go_id]

    def get_rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    with open(go_file) as f:
        for line in f:
            line = line.strip().replace("_", ":")
            if line.find("SubClassOf") == -1:
                continue
            left, right = line.split(" SubClassOf ")
            # C SubClassOf D
            if "and" not in line and "some" not in line and "SubClassOf" in line:
                go1, go2 = left, right
                nf1.append((get_index(go1), get_index(go2)))
            elif left.find(" and ") != -1:  # C and D SubClassOf E
                go1, go2 = left.split(" and ")
                go3 = right
                if "Nothing" not in go3:
                    nf2.append((get_index(go1), get_index(go2), get_index(go3)))
                else:
                    nf2_bot.append((get_index(go1), get_index(go2), get_index(go3)))
            elif left.find(" some ") != -1:  # R some C SubClassOf D
                rel, go1 = left.split(" some ")
                go2 = right
                nf4.append((get_rel_index(rel), get_index(go1), get_index(go2)))
            elif right.find(" some ") != -1:  # C SubClassOf R some D
                go1 = left
                rel, go2 = right.split(" some ")
                nf3.append((get_index(go1), get_rel_index(rel), get_index(go2)))
    return nf1, nf2, nf2_bot, nf3, nf4, relations, classes
