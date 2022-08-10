from torch.utils.data import Dataset


class Dataset1(Dataset):
    def __init__(self, tokenized_input, labels):
        self.input_ids = tokenized_input.input_ids
        self.token_type_ids = tokenized_input.token_type_ids
        self.attention_mask = tokenized_input.attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return (self.input_ids[index], self.token_type_ids[index], self.attention_mask[index]), self.labels[index]
