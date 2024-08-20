import argparse
import os

import torch

import data_loader.data_loaders as module_data
import model.priors as module_priors

from torch import nn
import model.model as module_arch
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
from parse_config import ConfigParser
from model.metric import accuracy

from fgvc.special.calibration import get_temperature, _ECELoss

from model.priors import MovingLocationPrior, BaseLocationPrior, MultipleHomeLocationsPrior, MultipleMovingLocationsPrior


def main(config, checkpoint_path):

    train_data_loader = config.init_obj('data_loader', module_data, training=True, num_workers=0)
    valid_data_loader = config.init_obj('data_loader', module_data, training=False, num_workers=0, shuffle=False)

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    model = config.init_obj('arch', module_arch, output_dim=valid_data_loader.num_classes)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Get prior definition (if any)
    prior = config.init_obj('prior', module_priors, dataset=train_data_loader.dataset) if 'prior' in config else None


    identity_labels = list(sorted(((k, v) for k, v in train_data_loader.dataset.name_to_identity_map.items()), key=lambda x: x[1]))
    identity_labels = [x[0] for x in identity_labels]




    with torch.no_grad():
        valid_outputs, valid_targets, valid_meta_data = [], [], []
        for i, (data, target, meta_data) in enumerate(tqdm(valid_data_loader, 'Processing validation set')):
            data, target = data.to(device), target.to(device)
            output = model(data)

            valid_outputs.append(output)
            valid_targets.append(target)
            valid_meta_data.extend(list(meta_data))

        if isinstance(valid_outputs[0], tuple):
            valid_outputs = tuple([torch.cat([v[i] for v in valid_outputs], dim=0).cpu() for i in range(len(valid_outputs[0]))])
        else:
            valid_outputs = torch.cat(valid_outputs, dim=0).cpu()

        valid_targets = torch.cat(valid_targets, dim=0).cpu()

    valid_meta_data = [json.loads(m) for m in valid_meta_data]

    for m, identity_id in zip(valid_meta_data, valid_targets):
        m['date'] = datetime.strptime(m['date'], '%d.%m.%Y %H:%M')
        m['is_location_in_training_set'] = m['grid_code'] in train_data_loader.dataset.identity_to_all_locations_map[identity_id]

    dates_sort = [m['date'] for m in valid_meta_data]
    image_id_sort = [m['image_id'].split('/')[-1] for m in valid_meta_data]
    sort_ix = np.lexsort((image_id_sort, dates_sort))

    if isinstance(valid_outputs, tuple):
        valid_outputs = tuple([v[sort_ix, ...] for v in valid_outputs])
    else:
        valid_outputs = valid_outputs[sort_ix, :]

    valid_targets = valid_targets[sort_ix]
    valid_meta_data = [valid_meta_data[ix] for ix in sort_ix]

    new_location_ix = [i for i, m in enumerate(valid_meta_data) if not m['is_location_in_training_set']]

    orig_predictions = valid_outputs[0]
    # orig_predictions = valid_outputs

    orig_acc = accuracy( orig_predictions, valid_targets)
    orig_acc_new_location = accuracy(orig_predictions[new_location_ix, :], valid_targets[new_location_ix])


    ece_criterion = _ECELoss()

    softplus = nn.Softplus()
    temperature = softplus(valid_outputs[1].squeeze()) + 1.
    logits = softplus(valid_outputs[0])
    calibrated_outputs = torch.softmax(logits / temperature[:, None], dim=1)
    orig_ece = ece_criterion(logits / temperature[:, None], valid_targets)

    print('Accuracy without prior {} (new location {}, ECE={})'.format(orig_acc * 100, orig_acc_new_location * 100, orig_ece))


    valid_predictions = prior.apply(calibrated_outputs, valid_meta_data)


    acc = accuracy( valid_predictions, valid_targets)
    acc_new_location = accuracy( valid_predictions[new_location_ix, :], valid_targets[new_location_ix])
    ece = ece_criterion(valid_predictions.log(), valid_targets)
    print('Accuracy with prior {} (new location {}, ECE={})'.format(acc * 100, acc_new_location * 100, ece))



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Foreground background')
    args.add_argument('-o', '--output_dir', type=str,
                      help='Output directory')

    args = args.parse_args()

    checkpoint_path = os.path.join(args.output_dir, 'model_best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    config = ConfigParser.from_file(os.path.join(args.output_dir, 'config.json'))

    main(config, checkpoint_path)