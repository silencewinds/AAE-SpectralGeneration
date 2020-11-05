import numpy as np
import torch.utils.data as data


class data_loader(data.Dataset):
    def __init__(self, file_name):
        super().__init__()
        self.m_data = np.load(file_name)
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._m_data = self.m_data
        print(self.m_data.shape)


    def __getitem__(self, item):
        return self.m_data[item]

    def __len__(self):
        return self.m_data.shape[0]

    def get_data(self):
        return self.m_data

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self.__len__():
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self.__len__() - start
            data_rest_part = self.m_data[start:self.__len__()]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.__len__())
                np.random.shuffle(perm)
                self._m_data = self.m_data[perm]
                self.m_data = self._m_data
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.m_data[start:end]
            return np.concatenate((data_rest_part, images_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.m_data[start:end]
