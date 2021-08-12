import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


test_file = open("./files/test.txt", 'r', encoding='utf-8')
valid_file = open("./files/valid.txt", 'r', encoding='utf-8')
test_iter = []
val_iter = []


line = test_file.readline()
while line:
    test_iter.append(line)
    line = test_file.readline()
test_file.close()


line = valid_file.readline()
while line:
    val_iter.append(line)
    line = valid_file.readline()
valid_file.close()


def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def tokenizer(line):
    line = line.lower()
    line = line.replace("\n", " ")
    return line.split()


# vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"], min_freq=5)
# vocab.set_default_index(vocab["<unk>"])
# torch.save(vocab, './files/turkishvocab.pt')
vocab = torch.load('./files/turkishvocab.pt')


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target


def train(epoch_nr):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch_nr, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def generate_sequence(mdl, src):
    src = src.unsqueeze(1)
    generate_step = 0
    while generate_step < 5:
        mask = torch.eye(len(src)).to(device)
        out = mdl.forward(src, mask)
        print(out[-1, :])
        out = torch.argmax(out[-1, :], dim=1)
        out = out.unsqueeze(0)
        src = torch.cat((src, out), dim=0)
        generate_step += 1
    src = src.squeeze(1)
    return src


val_data = data_process(val_iter)
test_data = data_process(test_iter)
batch_size = 10
eval_batch_size = 3
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
bptt = 35
ntokens = len(vocab)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
lr = 5.0  # learning rate
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
max_epoch = 10 # Epoch nr
file_nr = 41 # nr of training files
# model = torch.load('./files/turkishmodel.pt')
for epoch_nr in range(1,max_epoch+1):
    for training_file in range(1,file_nr+1):
        print("Epoch number: " + str(epoch_nr))
        print("Training data set: " + str(training_file))
        train_file = open(r"C:\Users\merih\Desktop\Sestek Yaz Stajı\dataset\train_" + str(training_file) + ".txt", 'r',
                          encoding='utf-8')
        train_iter = []
        line = train_file.readline()
        while line:
            train_iter.append(line)
            line = train_file.readline()
        train_file.close()
        train_data = data_process(train_iter)
        train_data = batchify(train_data, batch_size)
        best_val_loss = float("inf")
        best_model = None
        epoch_start_time = time.time()
        train(epoch_nr)
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch_nr, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
           best_val_loss = val_loss
           best_model = model

        torch.save(best_model, './files/turkishmodel.pt')

    scheduler.step()
