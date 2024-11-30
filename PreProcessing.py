import json
from tqdm import tqdm
from transformers import BertTokenizer

docred_rel2id = json.load(open('DocRED/DocRED_baseline_metadata/rel2id.json', 'r'))


def read_docred(file_in, tokenizer, max_seq_length=1024):
    """
        Reads a DOCRED dataset, processes the text, tokenizes the sentences, and returns features for model input.

        Args:
            file_in (str): Path to the input JSON file containing the DOCRED dataset.
                            The file should contain a list of samples, where each sample includes sentences and entities.
            tokenizer (object): A tokenizer object (e.g., from Hugging Face Transformers) that handles the tokenization process.
                                The tokenizer should have methods like `tokenize` and `convert_tokens_to_ids`.
            max_seq_length (int, optional): The maximum sequence length for tokenized input. Default is 1024.

        Returns:
            list: A list of feature dictionaries, where each dictionary contains the following keys:
                - 'input_ids': A list of token IDs representing the tokenized input text.
                - 'entity_pos': A list of tuples representing the start and end positions of entities in the tokenized text.
                - 'title': The title of the document from the sample.
    """
    i_line = 0
    features = []

    # Return None if the input file is empty
    if file_in == "":
        return None

    # Open the input file and load the JSON data
    with open(file_in, "r") as fh:
        data = json.load(fh)

    # Process each sample in the dataset
    for sample in tqdm(data, desc="Processing examples"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []

        # Identify the start and end positions of entities in the text
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0]))
                entity_end.append((sent_id, pos[1] - 1))

        # Tokenize each sentence and map entities to their tokenized positions
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        # Prepare entity positions (start, end) after tokenization
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end))

        # Process the sentences into input_ids
        sents = sents[:max_seq_length - 2]  # Trim to max sequence length
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        # Increment line counter
        i_line += 1

        # Prepare the feature for the current sample
        feature = {
            'input_ids': input_ids, # tokennized entities
            'entity_pos': entity_pos, # start and end position of the entities
            'title': sample['title'], # title
        }
        features.append(feature)

    # Print statistics
    print("# of documents {}".format(i_line))
    return features

if __name__ == '__main__':
    # Load the tokenizer (using BERT tokenizer as an example)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Path to the test file
    file_in = "DocRED/test.json"  # Update this path to your test.json location

    # Call the function
    features = read_docred(file_in, tokenizer, max_seq_length=1024)

    # Print the output
    for feature in features:
        print(feature)