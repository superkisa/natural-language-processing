import torch
import torch.nn as nn


import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.dropout = nn.Dropout(p=dropout)  # <YOUR CODE HERE>
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, src):
        # ? src = [src sent len, batch size]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # ? embedded = [src sent len, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded)
        # ? outputs = [src sent len, batch size, hid dim * n directions]
        # ? hidden = [n layers * n directions, batch size, hid dim]
        # ? cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, encoder_hid_dim, decoder_hid_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_hid_dim, decoder_hid_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = self.attn(encoder_outputs)

        # ? hidden: [batch, dec_hid_dim]
        # ? encoder_outputs_{q,b,d} * hidden_{d,b} = scores_{q,b}
        scores = torch.einsum("qbd,bd->qb", encoder_outputs, hidden[0])
        # Calculates pairwise dot product of `hidden` and each query element for batch
        scores = self.softmax(scores)
        return scores  # [query, batch]


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout_p = dropout

        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.attention = Attention(hid_dim, hid_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.out = nn.Linear(
            in_features=self.hid_dim * self.n_layers + self.hid_dim + self.emb_dim,
            out_features=output_dim,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        scores = self.attention(hidden, encoder_outputs)
        # ? encoder_outputs = [query, batch, enc_hid_dim]
        # ? scores          = [query, batch]
        # ? hidden          = [n layers * n directions, batch size, hid dim]
        # ? cell            = [n layers * n directions, batch size, hid dim]

        w_t = torch.einsum("qbe,qb->be", encoder_outputs, scores)
        # ? w_t             = [batch, query]
        w_t = w_t.unsqueeze(0)

        # n directions in the decoder will both always be 1, therefore:
        # ? hidden          = [n layers, batch size, hid dim]
        # ? context         = [n layers, batch size, hid dim]

        input = self.embedding(input.unsqueeze(0))

        w_t = w_t.squeeze(0)
        input = input.squeeze(0)
        hidden_flattened = hidden.permute(1, 2, 0).reshape(hidden.shape[1], -1)
        # [n_layers, n_batches, hid_dims] -> [n_batches, hid_dims * n_layers]
        # Flattens along layers and passes hidden state of each layer to self.out
        output = self.out(torch.cat((input, hidden_flattened, w_t), dim=-1))
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = trg[t] if teacher_force else top1

        return outputs
