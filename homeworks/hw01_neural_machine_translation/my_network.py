import torch
import torch.nn as nn
from icecream import ic
import random

ic.configureOutput(includeContext=True)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        # ? src =      [src sent len, batch size]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # ? embedded = [src sent len, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded)
        # ? outputs =  [src sent len, batch size, hid dim * 2]
        # ? hidden =   [n layers * 2, batch size, hid dim]
        # ? cell =     [n layers * 2, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )
        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))
        # ? outputs =  [src len, batch size, enc hid dim * n_directions]
        # ? hidden =   [batch size, dec hid dim]
        # ? cell =   [batch size, dec hid dim]

        return output, hidden.unsqueeze(0), cell.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # ? hidden =          [batch size, dec hid dim]
        # ? encoder_outputs = [src len, batch size, enc hid dim * 2]

        # repeat decoder hidden state src_len times
        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, encoder_outputs.shape[0], 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # ? hidden =          [batch size, src len, dec hid dim]
        # ? encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # ? energy =          [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # ? attention =       [batch size, src len]

        return nn.functional.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim

        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(
            input_size=(enc_hid_dim * 2) + emb_dim,
            hidden_size=dec_hid_dim,
        )

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_, hidden, cell, encoder_outputs):
        # ? input =           [batch size]
        # ? hidden =          [batch size, dec hid dim]
        # ? cell =            [batch size, dec hid dim]
        # ? encoder_outputs = [src len, batch size, enc hid dim * 2]

        input_ = input_.unsqueeze(0)
        # ? input =           [1, batch size]

        embedded = self.embedding(input_)
        embedded = self.dropout(embedded)
        # ? embedded =        [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)
        # ? a =               [batch size, src len]
        a = a.unsqueeze(1)
        # ? a =               [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # ? encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # ? weighted =        [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # ? weighted =        [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # ? rnn_input =        [1, batch size, (enc hid dim * 2) + emb dim]

        output, (hidden_dec, cell_dec) = self.rnn(rnn_input, (hidden, cell))
        # ? output = [seq len, batch size, dec hid dim * 1]
        # ? hidden = [n layers * 1, batch size, dec hid dim]
        # ? cell =   [n layers * 1, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # ? output = [1, batch size, dec hid dim]
        # ? hidden = [1, batch size, dec hid dim]
        # ? cell =   [1, batch size, dec hid dim]
        # this also means that output == hidden

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # ? prediction = [batch size, output dim]

        return prediction, hidden_dec, cell_dec


class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        enc_emb_dim,
        dec_emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        enc_n_layers=2,
        enc_dropout=0.5,
        dec_dropout=0.5,
        device=None,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            emb_dim=enc_emb_dim,
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim,
            n_layers=enc_n_layers,
            dropout=enc_dropout,
        )
        self.decoder = Decoder(
            output_dim=output_dim,
            emb_dim=dec_emb_dim,
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim,
            dropout=dec_dropout,
        )
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # ? src = [src sent len, batch size]
        # ? trg = [trg sent len, batch size]
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
        input_ = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input_ = trg[t] if teacher_force else top1

        return outputs
