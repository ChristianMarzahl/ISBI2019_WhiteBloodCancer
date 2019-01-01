import random
import re
import numpy as np

class SplitByPatient:

    def __init__(self,
                 hem_patients: dict,
                 all_patients: dict):

        self.hem_patients = hem_patients
        self.all_patients = all_patients
        self.error_margin = 0.1
        self.num_trails = 100

    def random_split(self, test_size=0.1, random_state=42):
        '''
        split files randomly
        same patient can be in train and val
        :param test_size:
        :param random_state:
        :return: hem train, all train, hem val, all val
        '''

        random.seed(random_state)

        hem_files = [self.hem_patients[key] for key in self.hem_patients.keys()]
        hem_files = [item for sublist in hem_files for item in sublist]  # flatten

        all_files = [self.all_patients[key] for key in self.all_patients.keys()]
        all_files = [item for sublist in all_files for item in sublist]  # flatten

        random.shuffle(hem_files)
        random.shuffle(all_files)

        hem_split_value = int(len(hem_files) * test_size)
        all_split_value = int(len(all_files) * test_size)

        return hem_files[hem_split_value:], \
               all_files[all_split_value:], \
               hem_files[:hem_split_value], \
               all_files[:all_split_value]

    def split_by_patients(self, test_size=0.1):
        '''
        split patients randomly
        files from the same patient can not be in train and val
        :param test_size:
        :return: hem train, all train, hem val, all val
        '''

        hem_files = [self.hem_patients[key] for key in self.hem_patients.keys()]
        hem_files = [item for sublist in hem_files for item in sublist]  # flatten

        all_files = [self.all_patients[key] for key in self.all_patients.keys()]
        all_files = [item for sublist in all_files for item in sublist]  # flatten

        hem_split_value = int(len(hem_files) * test_size)
        all_split_value = int(len(all_files) * test_size)

        hem_keys = list(self.hem_patients.keys())
        all_keys = list(self.all_patients.keys())

        hem_keys_train, hem_keys_val = self._split_by_patient(self.hem_patients, hem_keys, hem_split_value)
        all_keys_train, all_keys_val = self._split_by_patient(self.all_patients, all_keys, all_split_value)

        hem_files_train = [fn for fn in hem_files for key in hem_keys_train
                           if 'UID_{0}_'.format(key) in str(fn)]
        hem_files_val = [fn for fn in hem_files for key in hem_keys_val
                         if 'UID_{0}_'.format(key) in str(fn)]

        all_files_train = [fn for fn in all_files for key in all_keys_train
                           if 'UID_{0}_'.format(key) in str(fn)]
        all_files_val = [fn for fn in all_files for key in all_keys_val
                         if 'UID_{0}_'.format(key) in str(fn)]

        return hem_files_train, all_files_train, hem_files_val, all_files_val

    def split_by_regex(self, train_pat: re, val_pat: re):

        hem_files = [self.hem_patients[key] for key in self.hem_patients.keys()]
        hem_files = [item for sublist in hem_files for item in sublist]  # flatten

        all_files = [self.all_patients[key] for key in self.all_patients.keys()]
        all_files = [item for sublist in all_files for item in sublist]  # flatten

        hem_train = [fn for fn in hem_files if train_pat.search(str(fn)) is not None]
        hem_val = [fn for fn in hem_files if val_pat.search(str(fn)) is not None]

        all_train = [fn for fn in all_files if train_pat.search(str(fn)) is not None]
        all_val = [fn for fn in all_files if val_pat.search(str(fn)) is not None]

        return hem_train, all_train, hem_val, all_val

    def _split_by_patient(self, patients: dict, keys: list, split_value: int):
        count = 0
        selected_keys = []
        num_trails = 0
        while count < split_value:
            num_trails += 1
            selected_key = np.random.choice(keys, 1)[0]

            if count + len(patients[selected_key]) < split_value + split_value * self.error_margin:
                keys.remove(selected_key)
                selected_keys.append(selected_key)

                count += len(patients[selected_key])

                if num_trails > self.num_trails:
                    break

        return keys, selected_keys