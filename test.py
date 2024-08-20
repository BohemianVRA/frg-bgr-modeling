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


from model.priors import MovingLocationPrior, BaseLocationPrior, MultipleHomeLocationsPrior, MultipleMovingLocationsPrior
from utils.util import _ECELoss


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

    dataset_has_locations = hasattr( train_data_loader.dataset, 'identity_to_all_locations_map')

    for m, identity_id in zip(valid_meta_data, valid_targets):
        m[train_data_loader.dataset.date_column_name] = datetime.strptime(m[train_data_loader.dataset.date_column_name], '%d.%m.%Y %H:%M')
        if dataset_has_locations:
            m['is_location_in_training_set'] = m['grid_code'] in train_data_loader.dataset.identity_to_all_locations_map[identity_id]

    dates_sort = [m[train_data_loader.dataset.date_column_name] for m in valid_meta_data]
    image_id_sort = [m['image_id'].split('/')[-1] for m in valid_meta_data]
    sort_ix = np.lexsort((image_id_sort, dates_sort))

    if isinstance(valid_outputs, tuple):
        valid_outputs = tuple([v[sort_ix, ...] for v in valid_outputs])
    else:
        valid_outputs = valid_outputs[sort_ix, :]

    valid_targets = valid_targets[sort_ix]
    valid_meta_data = [valid_meta_data[ix] for ix in sort_ix]



    if isinstance(valid_outputs, tuple):
        logit_predictions = valid_outputs[0]
        temperature_predictions = valid_outputs[1]
    else:
        logit_predictions = valid_outputs
        temperature_predictions = None

    no_prior_acc = accuracy(logit_predictions, valid_targets)
    print('Accuracy without prior = {:.2f}%'.format(no_prior_acc * 100))

    if dataset_has_locations:
        new_location_ix = [i for i, m in enumerate(valid_meta_data) if not m['is_location_in_training_set']]
        no_prior_acc_new_location = accuracy(logit_predictions[new_location_ix, :], valid_targets[new_location_ix])
        print('Accuracy without prior in new locations only = {:.2f}%'.format(no_prior_acc_new_location * 100))

    if temperature_predictions is not None:

        # Calculate network outputs
        softplus = nn.Softplus()
        temperature = softplus(temperature_predictions.squeeze()) + 1.
        logits = softplus(logit_predictions)
        calibrated_outputs = torch.softmax(logits / temperature[:, None], dim=1)

        # Calculate ECE _before_ prior is applied
        ece_criterion = _ECELoss()
        no_prior_ece = ece_criterion(logits / temperature[:, None], valid_targets)
        print('Expected Calibration Error (ECE) without prior = {:.3f} '.format(no_prior_ece.item()))

        if prior is not None:
            # Apply prior
            valid_predictions = prior.apply(calibrated_outputs, valid_meta_data)

            acc = accuracy( valid_predictions, valid_targets)
            print('Accuracy with prior {} = {:.2f}%'.format(config['prior']['type'], acc * 100))

            if dataset_has_locations:
                acc_new_location = accuracy( valid_predictions[new_location_ix, :], valid_targets[new_location_ix])
                print('Accuracy with prior {} in new locations only = {:.2f}%'.format(config['prior']['type'], acc_new_location * 100))

            ece = ece_criterion(valid_predictions.log(), valid_targets)
            print('Expected Calibration Error (ECE) with prior {} = {:.3f}'.format(config['prior']['type'], ece.item()))


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