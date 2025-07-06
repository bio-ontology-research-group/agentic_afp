import click as ck
import torch as th
import numpy as np
from torch.nn import functional as F
from torch import optim
import math
from src.deepgo.torch_utils import FastTensorDataLoader
from src.deepgo.models import MLPModel
from src.deepgo.data import load_data
from src.deepgo.utils import propagate_annots
from src.ontology import Ontology
from multiprocessing import Pool
from functools import partial
from src.deepgo.metrics import compute_roc


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--model-name', '-m', type=ck.Choice([
        'mlp', 'mlp_esm']),
    default='mlp',
    help='Prediction model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test',]),
    help='Test data set name')
@ck.option(
    '--batch-size', '-bs', default=64,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(data_root, model_name, test_data_name, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.obo'
    model_files = {}
    out_file = f'{data_root}/{test_data_name}_predictions_{model_name}.pkl'
    for ont in ['mf', 'bp', 'cc']:
        model_files[ont] = f'{data_root}/{model_name}_{ont}.th'
        
    go = Ontology(go_file, with_rels=True)
    
    # Load the datasets
    features_length = 2560
    # test_data_file = f'{test_data_name}_data.pkl'
    test_data_file = 'test_data_diam_with_text.pkl'
    terms_dict, train_data, valid_data, test_data, test_df, genomes = load_data(
        data_root, features_length, test_data_file)
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    valid_labels_np = {}
    test_labels_np = {}
    for ont in ['mf', 'bp', 'cc']:
        valid_labels_np[ont] = valid_labels[ont].detach().cpu().numpy()
        test_labels_np[ont] = test_labels[ont].detach().cpu().numpy()

    nets = {}
    optimizers = {}
    for ont in ['mf', 'bp', 'cc']:
        nets[ont] = MLPModel(features_length, len(terms_dict[ont]), device).to(device)
        optimizers[ont] = optim.Adam(nets[ont].parameters(), lr=0.001)
    train_loader = FastTensorDataLoader(
        train_features, train_labels['mf'], train_labels['bp'], train_labels['cc'],
        batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        valid_features, valid_labels['mf'], valid_labels['bp'], valid_labels['cc'],
        batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        test_features, test_labels['mf'], test_labels['bp'], test_labels['cc'],
        batch_size=batch_size, shuffle=False)
    
    
    best_loss = {'mf': 10000.0, 'bp': 10000.0, 'cc': 10000.0}
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            nets['mf'].train()
            nets['bp'].train()
            nets['cc'].train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_features) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, mf_labels, bp_labels, cc_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    mf_labels = mf_labels.to(device)
                    mf_logits = nets['mf'](batch_features)
                    loss = {}
                    loss['mf'] = F.binary_cross_entropy(mf_logits, mf_labels)
                    bp_labels = bp_labels.to(device)
                    bp_logits = nets['bp'](batch_features)
                    loss['bp'] = F.binary_cross_entropy(bp_logits, bp_labels)
                    cc_labels = cc_labels.to(device)
                    cc_logits = nets['cc'](batch_features)
                    loss['cc'] = F.binary_cross_entropy(cc_logits, cc_labels)
                    for ont in ['mf', 'bp', 'cc']:
                        optimizer = optimizers[ont]
                        optimizer.zero_grad()
                        loss[ont].backward()
                        optimizer.step()
                        train_loss += loss[ont].detach().item()
            
            train_loss /= train_steps
            
            print('Validation')
            for ont in ['mf', 'bp', 'cc']:
                nets[ont].eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_features) / batch_size))
                valid_loss = {'mf': 0, 'bp': 0, 'cc': 0}
                preds = {'mf': [], 'bp': [], 'cc': []}
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, mf_labels, bp_labels, cc_labels in valid_loader:
                        bar.update(1)
                        labels = {'mf': mf_labels, 'bp': bp_labels, 'cc': cc_labels}
                        batch_features = batch_features.to(device)
                        for ont in ['mf', 'bp', 'cc']:
                            label = labels[ont].to(device)
                            logits = nets[ont](batch_features)
                            loss = F.binary_cross_entropy(logits, label)
                            valid_loss[ont] += loss.detach().item()
                            preds[ont].append(logits.detach().cpu().numpy())
                for ont in ['mf', 'bp', 'cc']:
                    valid_loss[ont] /= valid_steps
                    preds[ont] = np.concatenate(preds[ont])
                    roc_auc = compute_roc(valid_labels_np[ont], preds[ont])
                    print(f'Ontology {ont}: AUC - {roc_auc}, Loss - {valid_loss[ont]}')
                    if valid_loss[ont] < best_loss[ont]:
                        best_loss[ont] = valid_loss[ont]
                        print(f'Saving model {ont}')
                        th.save(nets[ont].state_dict(), model_files[ont])

    for ont in ['mf', 'bp', 'cc']:
        # Loading best model    
        print('Loading the best model')
        nets[ont].load_state_dict(th.load(model_files[ont]))
        nets[ont].eval()
        with th.no_grad():
            test_steps = int(math.ceil(len(test_features) / batch_size))
            test_loss = 0
            preds = []
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, mf_labels, bp_labels, cc_labels in test_loader:
                    batch_labels = {'mf': mf_labels, 'bp': bp_labels, 'cc': cc_labels}
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels[ont].to(device)
                    logits = nets[ont](batch_features)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds.append(logits.detach().cpu().numpy())
                test_loss /= test_steps
            preds = np.concatenate(preds)
            roc_auc = compute_roc(test_labels_np[ont], preds)
            print(f'Test Loss {ont} - {test_loss}, AUC - {roc_auc}')

        preds = list(preds)
        # Propagate scores using ontology structure
        with Pool(32) as p:
            preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict[ont]), preds)

        test_df[f'{ont}_preds'] = preds

    test_df.to_pickle(out_file)
    

if __name__ == '__main__':
    main()
