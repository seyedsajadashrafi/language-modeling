import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import config


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]

    # hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            hidden = model.init_hidden_states(1, device)
            prediction = model(src, hidden)
            # prediction = torch.argmax(prediction[:, -1], dim=-1)
            # print(prediction)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            # if prediction == vocab['<eos>']:
            if prediction == vocab['.']:
                break

            indices.append(prediction)

        itos = vocab.get_itos()
        tokens = [itos[i] for i in indices]
        return tokens


if __name__ == '__main__':
    # load model
    model_path = 'model-ppl_141.5.pt'
    model = torch.load(model_path)
    model.eval()

    # load tokenizer
    tokenizer = get_tokenizer('basic_english')

    # load vocab
    vocab = torch.load('vocab.pt')

    prompt = 'in the'
    max_seq_len = 70
    seed = 0

    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, tokenizer,
                              vocab, config.device, seed)
        print(str(temperature) + '\n' + ' '.join(generation) + '\n')