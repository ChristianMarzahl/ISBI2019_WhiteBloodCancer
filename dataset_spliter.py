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

    def split_by_folds(self, folds: int = 5):

        hem_sorted = np.asarray(sorted(self.hem_patients.items(), key=lambda kv: len(kv[1])))
        all_sorted = np.asarray(sorted(self.all_patients.items(), key=lambda kv: len(kv[1])))

        hem_count = len(self.hem_patients)
        all_count = len(self.all_patients)
        step = folds * 2

        fold_keys_hem = {}
        fold_keys_all = {}
        for i in range(folds):
            fold_keys_hem[i] = list(range(i, hem_count, step)) \
                               + list(range(step - (i + 1), hem_count, step))
            fold_keys_all[i] = list(range(i, all_count, step)) \
                               + list(range(step - (i + 1), all_count, step))

        flatten = lambda l: [item for sublist in l for item in sublist]
        fold_files = {}
        for i in range(folds):
            fold_files[i] = flatten(hem_sorted[fold_keys_hem[i]][:, 1]) \
                            + flatten(all_sorted[fold_keys_all[i]][:, 1])

        return fold_files

    def split_by_patients(self, test_size=0.1, return_keys: bool = False):
        '''
        split patients randomly
        files from the same patient can not be in train and val
        :param return_keys:
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

        if return_keys:
            return hem_keys_val + all_keys_val
        else:
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

    def split_by_num_patients(self, fnames:list, num_all: int=5, num_hem: int=3):

        hem_val_pat_keys = list(np.random.choice(list(self.hem_patients.keys()), 2 * num_hem, replace=False))
        all_val_pat_keys = list(np.random.choice(list(self.all_patients.keys()), 2 * num_all, replace=False))

        fold_0 = hem_val_pat_keys[:num_hem] + all_val_pat_keys[:num_all]
        fold_1 = hem_val_pat_keys[num_hem:] + all_val_pat_keys[num_all:]

        val_fold_0 = []
        val_fold_1 = []
        for id, fn in enumerate(fnames):
            for pt in fold_0:
                if 'UID_{}_'.format(pt) in str(fn.stem):
                    val_fold_0.append(id)

            for pt in fold_1:
                if 'UID_{}_'.format(pt) in str(fn.stem):
                    val_fold_1.append(id)

        train_fold_0 = []
        train_fold_1 = []
        for id, fn in enumerate(fnames):
            if id not in val_fold_0:
                train_fold_0.append(id)

            if id not in val_fold_1:
                train_fold_1.append(id)

        return [train_fold_0, train_fold_1], [val_fold_0, train_fold_1], hem_val_pat_keys+all_val_pat_keys






