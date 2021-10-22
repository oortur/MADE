import os
from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
from torchtext.legacy.data import BucketIterator, TabularDataset
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def prepare_split_datasets(path_to_data, path_split_data, train_size=0.8, val_size=0.05):
    txt = []
    with open(path_to_data, "r") as f_in:
        txt = [line for line in f_in.readlines()]

    txt = np.array(txt)
    size = len(txt)

    indexes = np.random.permutation(np.arange(size))
    train_ix = indexes[: int(train_size * size)]
    val_ix = indexes[int(train_size * size) : int((train_size + val_size) * size)]
    test_ix = indexes[int((train_size + val_size) * size) :]
    
    os.makedirs(path_split_data, exist_ok=True)
    with open(os.path.join(path_split_data, "train_data.txt") ,"w") as f_out:
        f_out.write(''.join(txt[train_ix]))
    with open(os.path.join(path_split_data, "val_data.txt") ,"w") as f_out:
        f_out.write(''.join(txt[val_ix]))
    with open(os.path.join(path_split_data, "test_data.txt") ,"w") as f_out:
        f_out.write(''.join(txt[test_ix]))


def prepare_iterators(
    src_field, trg_field, batch_size, device, path_to_data,
    train_path=None, val_path=None, test_path=None,
):
    if path_to_data is not None:
        dataset = TabularDataset(
            path=path_to_data,
            format='tsv',
            fields=[('trg', trg_field), ('src', src_field)],
        )

        train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])

    else:
        train_data = TabularDataset(
            path=train_path,
            format='tsv',
            fields=[('trg', trg_field), ('src', src_field)],
        )
        valid_data = TabularDataset(
            path=val_path,
            format='tsv',
            fields=[('trg', trg_field), ('src', src_field)],
        )
        test_data = TabularDataset(
            path=test_path,
            format='tsv',
            fields=[('trg', trg_field), ('src', src_field)],
        )

    src_field.build_vocab(train_data, min_freq=3)
    trg_field.build_vocab(train_data, min_freq=3)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=batch_size, 
        device=device,
        sort_key=lambda x: len(x.src),
    )

    # print statistics
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}\n")
    print(f"Unique tokens in source (ru) vocabulary: {len(src_field.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(trg_field.vocab)}")

    return (src_field, trg_field), (train_data, valid_data, test_data), (train_iterator, valid_iterator, test_iterator)


def show_length_distributions(src_length, trg_length):
    plt.figure(figsize=[8, 4])

    plt.subplot(1, 2, 1)
    plt.title("source length")
    plt.hist(list(src_length), bins=20);

    plt.subplot(1, 2, 2)
    plt.title("translation length")
    plt.hist(list(trg_length), bins=20);
    
    plt.show();


def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def decoder_type(model):
    str_type = str(type(model.decoder))
    if "AttentionDecoder" in str_type:
        return "transformer"
    elif "ConvDecoder" in str_type:
        return "cnn"
    elif "LSTMDecoder" in str_type:    
        return "rnn"
    elif "GRUAttnDecoder" in str_type:    
        return "rnn"
    return "other"


def train_epoch(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None):

    decoder = decoder_type(model)
    assert decoder in ["transformer", "cnn", "rnn"], 'model.decoder must be one of "AttentionDecoder", "ConvDecoder", "LSTMDecoder"'
   
    model.train()
    output_dim = model.decoder.output_dim
    epoch_loss = 0
    history = []

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg
        bs = src.shape[1]
        
        optimizer.zero_grad()

        if decoder in ["transformer", "cnn"]:
            output = model(src, trg[:-1]) #output = [trg_len - 1, bs, output_dim]
            output = output.contiguous().view(-1, output_dim) #output = [bs * (trg_len - 1), output_dim]
        else:
            output = model(src, trg) #output = [trg_len, bs, output_dim]
            output = output[1:].view(-1, output_dim) #output = [bs * (trg_len - 1), output_dim]
        
        trg = trg[1:].contiguous().view(-1) #trg = [bs * (trg_len - 1)]        

        loss = criterion(output, trg)
        loss.backward()
                
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy())
        if (i + 1)%(1000 // bs) == 0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()

    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, criterion):

    decoder = decoder_type(model)
    assert decoder in ["transformer", "cnn", "rnn"], 'model.decoder must be one of "AttentionDecoder", "ConvDecoder", "LSTMDecoder"'

    model.eval()
    output_dim = model.decoder.output_dim
    epoch_loss = 0
    history = []

    with torch.no_grad():
        
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            if decoder in ["transformer", "cnn"]:
                output = model(src, trg[:-1]) #output = [trg_len - 1, bs, output_dim]
                output = output.contiguous().view(-1, output_dim) #output = [bs * (trg_len - 1), output_dim]
            else:
                output = model(src, trg) #output = [trg_len, bs, output_dim]
                output = output[1:].view(-1, output_dim) #output = [bs * (trg_len - 1), output_dim]

            trg = trg[1:].contiguous().view(-1) #trg = [bs * (trg_len - 1)]        

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def report_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_batch_translation(src, trg, model, TRG, device="cuda", bpe_model=None, max_len=100, print_result=True):

    decoder = decoder_type(model)
    assert decoder in ["transformer", "cnn", "rnn"], 'model.decoder must be one of "AttentionDecoder", "ConvDecoder", "LSTMDecoder"'
    
    model.eval()
    original = []
    generated = []

    if decoder == "rnn":
        output = model(src, trg, 0) #turn off teacher forcing
        output = output.argmax(dim=-1)
        generated = output[1:].cpu().numpy().T

    elif decoder == "transformer":
        src = src.permute(1, 0)
        src_mask = model.make_src_mask(src) 
        
        with torch.no_grad():
            if "BERTEncoder" in str(type(model.encoder)): 
                enc_src = model.encoder(src)
            else:
                enc_src = model.encoder(src, src_mask)

        bs = src.shape[0]
        gen_tensor = torch.LongTensor([TRG.vocab.stoi[TRG.init_token]]).repeat(bs, 1).to(device)

        for t in range(1, max_len):
            trg_mask = model.make_trg_mask(gen_tensor)
            with torch.no_grad():
                output = model.decoder(gen_tensor, enc_src, trg_mask, src_mask) 
            pred_indexes = output[:, -1, :].argmax(dim=1)
            gen_tensor = torch.hstack([gen_tensor, pred_indexes.unsqueeze(1)])
            # gen_tensor = [bs, t+1]
        
        generated = gen_tensor.cpu().numpy()

    elif decoder == "cnn":
        src = src.permute(1, 0)
        with torch.no_grad():
            encoder_conved, encoder_combined = model.encoder(src)

        bs = src.shape[0]
        gen_tensor = torch.LongTensor([TRG.vocab.stoi[TRG.init_token]]).repeat(bs, 1).to(device)

        for t in range(1, max_len):
            with torch.no_grad():
                output = model.decoder(gen_tensor, encoder_conved, encoder_combined)
            pred_indexes = output[:, -1, :].argmax(dim=1)
            gen_tensor = torch.hstack([gen_tensor, pred_indexes.unsqueeze(1)])
            # gen_tensor = [bs, t+1]

        generated = gen_tensor.cpu().numpy()
    
    original = trg.permute(1, 0).cpu().numpy()

    original = [get_text(orig.tolist(), TRG.vocab) for orig in original]
    generated = [get_text(gen.tolist(), TRG.vocab) for gen in generated]

    if bpe_model is not None:
        original = [bpe_model.decode([bpe_model.subword_to_id(tok) for tok in orig])[0].split() for orig in original]
        generated = [bpe_model.decode([bpe_model.subword_to_id(tok) for tok in gen])[0].split() for gen in generated]

    if print_result:
        for orig, gen in zip(original, generated):
            # print(f"Original: {orig}\nGenerated: {gen}\n")
            print('Original: {}'.format(' '.join(orig)))
            print('Generated: {}'.format(' '.join(gen)))
            print()
    
    return original, generated


def calculate_bleu(model, test_iterator, TRG, device="cuda", max_len=100, bpe_model=None, verbose=True):
    original_text = []
    generated_text = []
    n_shorter_4 = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_iterator):
            # , display=verbose
            orig, gen = generate_batch_translation(batch.src, batch.trg, model, TRG=TRG, device=device, bpe_model=bpe_model, max_len=max_len, print_result=False)
            # for i, sentence in enumerate(gen):
            #     if len(sentence[0].split()) < 4:
            #         n_shorter_4 += 1
            #     else:
            #         original_text.append(orig[i])
            #         generated_text.append(gen[i])
            original_text.extend(orig)
            generated_text.extend(gen)
    
    # print(f"Number of translations shorter than 4: {n_shorter_4}")
    return corpus_bleu([[orig] for orig in original_text], generated_text)
    # , smoothing_function=SmoothingFunction().method1)


def infer_mask(seq, eos_ix, batch_first=True, include_eos=True, dtype=torch.float):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if batch_first else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: lengths, int32 vector of shape [batch]
    """
    assert seq.dim() == 2
    is_eos = (seq == eos_ix).to(dtype=torch.float)
    if include_eos:
        if batch_first:
            is_eos = torch.cat((is_eos[:,:1]*0, is_eos[:, :-1]), dim=1)
        else:
            is_eos = torch.cat((is_eos[:1,:]*0, is_eos[:-1, :]), dim=0)
    count_eos = torch.cumsum(is_eos, dim=1 if batch_first else 0)
    mask = count_eos == 0
    return mask.to(dtype=dtype)


def infer_length(seq, eos_ix, batch_first=True, include_eos=True, dtype=torch.long):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :param include_eos: if True, the time-step where eos first occurs is has mask = 1
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    mask = infer_mask(seq, eos_ix, batch_first, include_eos, dtype)
    return torch.sum(mask, dim=1 if batch_first else 0)


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.to(dtype=torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=y.device).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def translate(model, src, greedy=True, eps=1e-10):
    # src = [bs, src_len]

    src_mask = model.make_src_mask(src)
    enc_src = model.encoder(src, src_mask)

    bs = src.shape[0]
    trg_vocab_size = model.decoder.output_dim

    generated = torch.LongTensor([model.trg_init_idx]).repeat(bs, 1).to(model.device)
    logits_seq = [torch.log(to_one_hot(generated[:,0], trg_vocab_size) + eps)]

    for t in range(1, model.decoder.output_max_length):
        trg_mask = model.make_trg_mask(generated)
        
        logits = model.decoder(generated, enc_src, trg_mask, src_mask)[:, -1, :]
        # logits = [bs, output_dim]
        if greedy:
            gen_t = logits.argmax(dim=1)
        else:
            probs = F.softmax(logits, dim=1)
            gen_t = torch.multinomial(probs, 1)[:, 0]

        generated = torch.hstack([generated, gen_t.unsqueeze(1)])
        # gen_tensor = [bs, t+1]

        logits_seq.append(logits)

    # generated = [bs, output_dim], log_probs = [bs, max_len, output_dim]
    return generated, F.log_softmax(torch.stack(logits_seq, 1), dim=-1)


def compute_reward(trg, translations, trg_vocab):
    """ computes sample-wise reward given token ids for inputs and translations """
    # trg = [bs, trg_len]
    # translations = [bs, max_len]
    original_text = [get_text(orig.tolist(), trg_vocab) for orig in trg]
    generated_text = [get_text(gen.tolist(), trg_vocab) for gen in translations]

    # calculate reward as mean of independent BLEU scores of order 2, 3, and 4 
    reward = 0
    for N in range(2, 4+1):
        bleu_n = bleu_score(generated_text, [[orig] for orig in original_text], max_n=N, weights=(1/N,)*N)
        reward += 1/3 * bleu_n

    # print(f'BLEU: {reward}')

    return torch.tensor([reward])


def scst_objective_on_batch(model, src, trg, trg_field, device="cuda"):
    """ Compute pseudo-loss for policy gradient given a batch of sources """
    # src = [bs, src_len]
    # trg = [bs, trg_len]

    # make sample and greedy translation for input src

    sample_translations, sample_logp = translate(model, src, greedy=False)
    greedy_translations, greedy_logp = translate(model, src, greedy=True)
    # sample_translations = greedy_translations = [bs, max_len]
    # sample_logp = greedy_logp = [bs, max_len, output_dim]

    # compute rewards and advantage
    # be careful with the device, rewards require casting to numpy, so send everything to cpu
    rewards = compute_reward(trg.cpu(), sample_translations.cpu(), trg_field.vocab)
    baseline = compute_reward(trg.cpu(), greedy_translations.cpu(), trg_field.vocab)

    # compute advantage using rewards and baseline
    # be careful with the device, advantage is used to compute gradients, so send it to device
    advantage = (rewards - baseline).float().to(device)

    # compute log_pi(a_t|s_t), shape = [batch, seq_length]
    logp_sample = torch.sum(to_one_hot(sample_translations, n_dims=len(trg_field.vocab)) * sample_logp, dim=-1)

    # policy gradient pseudo-loss. Gradient of J is exactly policy gradient.
    J = logp_sample * advantage[:, None]
    assert J.dim() == 2, "please return elementwise objective, don't compute mean just yet"

    # average with mask
    mask = infer_mask(sample_translations, trg_field.vocab.stoi[trg_field.eos_token])
    loss = - torch.sum(J * mask) / torch.sum(mask)

    # regularize with negative entropy. Don't forget the sign!
    # note: for entropy you need probabilities for all tokens (sample_logp), not just logp_sample
    entropy = -torch.sum(torch.exp(sample_logp) * sample_logp, dim=-1)
    # <compute entropy matrix of shape[batch, seq_length], H = -sum(p*log_p), don't forget the sign!>
    # hint: you can get sample probabilities from sample_logp using math :)

    assert entropy.dim() == 2, "please make sure elementwise entropy is of shape [batch,time]"
    reg = - 0.1 * torch.sum(entropy * mask) / torch.sum(mask)

    return loss + reg, torch.sum(entropy * mask) / torch.sum(mask)
