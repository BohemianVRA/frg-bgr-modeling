import copy
import math
import torch

def parse_coords(coord_string):
    coord = coord_string.split('-')
    assert len(coord) == 2
    return int(coord[0]), int(coord[1])

def grid_distance(a, b):
    a, b = parse_coords(a), parse_coords(b)

    dist = math.floor(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    return dist


def date_distance(timestamp, year):
    diff = abs(timestamp.year - year)
    return diff


def calculate_prior(sample_meta_data, prior_map, num_classes, column_name, eps):
    prior = torch.zeros(num_classes)
    metadata_value = sample_meta_data[column_name]

    for class_id, class_id_priors in enumerate(prior_map):
        if metadata_value in class_id_priors:
            prior[class_id] = class_id_priors[metadata_value] / class_id_priors['Total'] + eps
        else:
            prior[class_id] = eps

    prior /= prior.sum()

    return prior

class BaseLocationPrior(object):
    def __init__(self, dataset, alpha, threshold=0):
        self.identity_to_base_location_map = dataset.identity_to_base_location_map
        self.alpha = alpha
        self.threshold = threshold


    def apply(self, appearance_prob, valid_meta_data):

        base_location_dist = torch.zeros(appearance_prob.size())

        for sample_ix, sample_meta_data in enumerate(valid_meta_data):
            location = sample_meta_data['grid_code']
            for identity_id, base_location in enumerate(self.identity_to_base_location_map):
                base_location_dist[sample_ix, identity_id] = max(grid_distance(location, base_location[0][0]) - self.threshold, 0)

        base_location_prob = self.alpha * torch.exp(-base_location_dist * self.alpha)

        prob = appearance_prob * base_location_prob.to(appearance_prob.device)

        return prob



class MultipleHomeLocationsPrior(object):
    def __init__(self, dataset, alpha, threshold=0):
        self.identity_to_base_location_map = dataset.identity_to_base_location_map
        self.alpha = alpha
        self.threshold = threshold


    def apply(self, appearance_prob, valid_meta_data):

        base_location_dist = torch.zeros(appearance_prob.size())

        for sample_ix, sample_meta_data in enumerate(valid_meta_data):
            location = sample_meta_data['grid_code']
            for identity_id, base_locations in enumerate(self.identity_to_base_location_map):
                min_distance = 1000
                for base_location in base_locations:
                    min_distance = min(max(grid_distance(location, base_location[0]) - self.threshold, 0), min_distance)

                base_location_dist[sample_ix, identity_id] = min_distance

        base_location_prob = self.alpha * torch.exp(-base_location_dist * self.alpha)

        prob = appearance_prob * base_location_prob.to(appearance_prob.device)

        return prob


class MovingLocationPrior(object):
    def __init__(self, dataset, alpha, location_update_prob_threshold=0.5):
        self.identity_to_base_location_map = dataset.identity_to_base_location_map
        self.alpha = alpha
        self.location_update_prob_threshold = location_update_prob_threshold

    def apply(self, appearance_prob, valid_meta_data):

        identity_to_base_location_map = copy.deepcopy(self.identity_to_base_location_map)
        identity_to_base_location_map = [loc[0][0] for loc in identity_to_base_location_map]

        num_identities = appearance_prob.size(1)

        previous_positions = {k: [] for k in range(num_identities)}


        prob = torch.zeros(appearance_prob.size())

        for sample_ix, sample_meta_data in enumerate(valid_meta_data):
            location = sample_meta_data['grid_code']

            base_location_dist = torch.zeros(num_identities)

            for identity_id, base_location in enumerate(identity_to_base_location_map):
                base_location_dist[identity_id] = max(grid_distance(location, base_location), 0)

            base_location_prob = torch.exp(-base_location_dist * self.alpha)
            base_location_prob = base_location_prob / base_location_prob.sum()

            prob[sample_ix, :] = appearance_prob[sample_ix, :] * base_location_prob
            prob[sample_ix, :] = prob[sample_ix, :] / prob[sample_ix, :].sum()

            pred_label = torch.argmax(prob[sample_ix, :]).item()

            pred_label_prob = prob[sample_ix, pred_label] .item()

            previous_positions[pred_label].append(location)

            if pred_label_prob > self.location_update_prob_threshold:
                identity_to_base_location_map[pred_label] = location








        return prob.to(appearance_prob.device)


class MultipleMovingLocationsPrior(object):
    def __init__(self, dataset, alpha, observation_count_update_threshold=2):
        self.identity_to_base_location_map = dataset.identity_to_base_location_map
        self.alpha = alpha
        self.observation_count_update_threshold = observation_count_update_threshold


    def apply(self, appearance_prob, valid_meta_data):

        identity_to_base_location_map = copy.deepcopy(self.identity_to_base_location_map)

        num_identities = appearance_prob.size(1)

        previous_positions = {k: [] for k in range(num_identities)}


        prob = torch.zeros(appearance_prob.size())

        for sample_ix, sample_meta_data in enumerate(valid_meta_data):
            location = sample_meta_data['grid_code']

            base_location_dist = torch.zeros(num_identities)

            for identity_id, base_locations in enumerate(identity_to_base_location_map):
                min_distance = 1000
                for base_location in base_locations:
                    min_distance = min(max(grid_distance(location, base_location[0]), 0), min_distance)

                base_location_dist[identity_id] = min_distance



            base_location_prob = self.alpha * torch.exp(-base_location_dist * self.alpha)

            prob[sample_ix, :] = appearance_prob[sample_ix, :] * base_location_prob

            pred_label = torch.argmax(prob[sample_ix, :]).item()

            previous_positions[pred_label].append(location)

            if len(previous_positions[pred_label]) >= self.observation_count_update_threshold:
                locations = previous_positions[pred_label][:self.observation_count_update_threshold]

                # Update location
                if all([locations[0] == loc for loc in locations]):
                    identity_to_base_location_map[pred_label].append((location, 1))







        return prob.to(appearance_prob.device)




class TimeDecayPrior(object):
    def __init__(self, dataset, alpha, threshold=0, year_update_threshold=1):
        self.identity_to_last_year_map = dataset.identity_to_last_year_map
        self.alpha = alpha
        self.threshold = threshold
        self.year_update_threshold = year_update_threshold


    def apply(self, appearance_prob, valid_meta_data):

        identity_to_last_year_map = copy.deepcopy(self.identity_to_last_year_map)
        num_identities = appearance_prob.size(1)

        previous_years = {k: [] for k in range(num_identities)}


        prob = torch.zeros(appearance_prob.size())

        for sample_ix, sample_meta_data in enumerate(valid_meta_data):
            timestamp = sample_meta_data['timestamp']

            last_year_dist = torch.zeros(num_identities)

            for identity_id, last_year in enumerate(identity_to_last_year_map):
                last_year_dist[identity_id] = max(date_distance(timestamp, last_year) - self.threshold, 0)

            base_location_prob = self.alpha * torch.exp(-last_year_dist * self.alpha)

            prob[sample_ix, :] = appearance_prob[sample_ix, :] * base_location_prob

            pred_label = torch.argmax(prob[sample_ix, :]).item()

            previous_years[pred_label].append(timestamp.year)

            if len(previous_years[pred_label]) > self.year_update_threshold:
                locations = previous_years[pred_label][:(self.year_update_threshold + 1)]

                # Update location
                if all([locations[0] == loc for loc in locations]):
                    identity_to_last_year_map[pred_label] = timestamp.year







        return prob.to(appearance_prob.device)



