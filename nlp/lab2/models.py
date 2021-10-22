import random

import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt



# ----- RNN models -----

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim,
        )
            # <YOUR CODE HERE>
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        #src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)
        
        embedded = self.dropout(embedded)
        
        output, (hidden, cell) = self.rnn(embedded)
        #embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell
    

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim,
        )
        
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
        )
        
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, cell):
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #sent len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.out(output.squeeze(0))
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class LSTMSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs



# ----- RNN + Attention models -----

class GRUEncoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        emb_dim, 
        enc_hid_dim, 
        dec_hid_dim, 
        dropout,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [src_len, bs]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src_len, bs, emb_dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src_len, bs, hid _im * num_directions]
        #hidden = [n_layers * num_directions, bs, hid_dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src_len, bs, enc_hid_dim * 2]
        #hidden = [bs, dec_hid_dim]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(
        self, 
        enc_hid_dim, 
        dec_hid_dim,
    ):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [bs, dec_hid_dim]
        #encoder_outputs = [src_len, bs, enc_hid_dim * 2]
        
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [bs, src_len, dec_hid_dim]
        #encoder_outputs = [bs, src_len, enc_hid_dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [bs, src_len, dec_hid_dim]

        attention = self.v(energy).squeeze(2)
        #attention= [bs, src_len]
        
        return F.softmax(attention, dim=1)


class GRUAttnDecoder(nn.Module):
    def __init__(
        self, 
        output_dim, 
        emb_dim, 
        enc_hid_dim, 
        dec_hid_dim, 
        dropout, 
        attention,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #input = [bs]
        #hidden = [bs, dec_hid_dim]
        #encoder_outputs = [src_len, bs, enc_hid_dim * 2]
        
        input = input.unsqueeze(0)
        #input = [1, bs]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, bs, emb_dim]
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        #a = [bs, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [bs, src_len, enc_hid_dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [bs, 1, enc_hid_dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, bs, enc_hid_dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, bs, (enc_hid_dim * 2) + emb_dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [seq_len, bs, dec_hid_dim * n_directions]
        #hidden = [n_layers * n_directions, bs, dec_hid_dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, bs, dec_hid_dim]
        #hidden = [1, bs, dec_hid_dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [bs, output_dim]
        
        return prediction, hidden.squeeze(0)


class GRUAttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src_len, bs]
        #trg = [trg_len, bs]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        trg_len, bs = trg.shape
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, bs, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs



# ----- CNN models -----

class ConvEncoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        emb_dim, 
        hid_dim, 
        n_layers, 
        kernel_size, 
        dropout, 
        device,
        max_length=100,
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels = hid_dim, 
            out_channels = 2 * hid_dim, 
            kernel_size = kernel_size, 
            padding = (kernel_size - 1) // 2,
        ) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        #src = [bs, src_len]
        bs, src_len = src.shape
        
        #create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(bs, 1).to(self.device)
        #pos = [bs, src_len]
        
        #combine token and positional embeddings by elementwise summing
        embedded = self.dropout(self.tok_embedding(src) + self.pos_embedding(pos))
        #embedded = [bs, src_len, emb_dim]
        
        #pass embedded through linear layer to convert from emb_dim to hid_dim
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1) 
        #conv_input = [bs, hid_dim, src_len]
        
        #begin convolutional blocks...
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            #conved = [bs, 2 * hid_dim, src_len]

            conved = F.glu(conved, dim = 1)
            #conved = [bs, hid_dim, src_len]
            
            conved = (conved + conv_input) * self.scale
            #conved = [bs, hid_dim, src_len]
            
            conv_input = conved
        #...end convolutional blocks
        
        #permute and convert back to emb_dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        #conved = [bs, src_len, emb_dim]
        
        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        #combined = [bs, src_len, emb_dim]
        
        return conved, combined


class ConvDecoder(nn.Module):
    def __init__(
        self, 
        output_dim, 
        emb_dim, 
        hid_dim, 
        n_layers, 
        kernel_size, 
        dropout, 
        trg_pad_idx, 
        device,
        max_length=100,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads=1, dropout=dropout)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels = hid_dim, 
            out_channels = 2 * hid_dim, 
            kernel_size = kernel_size,
        ) for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
      
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        #embedded = [bs, trg_len, emb_dim]
        #conved = [bs, hid_dim, trg_len]
        #encoder_conved = [bs, src_len, emb_dim]
        #encoder_combined = [bs, src_len, emb_dim]
        
        #permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        #conved_emb = [bs, trg_len, emb_dim]
        
        combined = (conved_emb + embedded) * self.scale
        #combined = [bs, trg_len, emb_dim]
        
        #
        attended_encoding, _ = self.attn(
            query=combined.permute(1, 0, 2),
            key=encoder_conved.permute(1, 0, 2),
            value=encoder_combined.permute(1, 0, 2),
        )
        attended_encoding = attended_encoding.permute(1, 0, 2)
        #attended_encoding = [bs, trg_len, emd_dim]

        #convert from emb_dim -> hid_dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        #attended_encoding = [bs, trg_len, hid_dim]
        
        #apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        #attended_combined = [bs, hid_dim, trg_len]
        
        return attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
        #trg = [bs, trg_len]
        #encoder_conved = [bs, src_len, emb_dim]
        #encoder_combined = [bs, src_len, emb_dim]
        bs, trg_len = trg.shape
            
        #create position tensor
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(bs, 1).to(self.device)
        #pos = [bs, trg_len]
        
        #combine token and positional embeddings by elementwise summing
        embedded = self.dropout(self.tok_embedding(trg) + self.pos_embedding(pos))
        #embedded = [bs, trg_len, emb_dim]
        
        #pass embedded through linear layer to go through emb_dim -> hid_dim
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1) 
        #conv_input = [bs, hid_dim, trg_len]
        
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
        
            #apply dropout
            conv_input = self.dropout(conv_input)
        
            #need to pad so decoder can't "cheat"
            padding = torch.zeros(bs, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
                
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)
            #padded_conv_input = [bs, hid_dim, trg_len + kernel_size - 1]
        
            #pass through convolutional layer
            conved = conv(padded_conv_input)
            #conved = [bs, 2 * hid_dim, trg_len]
            
            conved = F.glu(conved, dim = 1)
            #conved = [bs, hid_dim, trg_len]
            
            #calculate attention
            conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)
            #attention = [bs, trg_len, src_len]
            
            #apply residual connection
            conved = (conved + conv_input) * self.scale
            #conved = [bs, hid_dim, trg_len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))
        #conved = [bs, trg_len, emb_dim]
            
        output = self.fc_out(self.dropout(conved))
        #output = [bs, trg_len, output_dim]
            
        return output


class ConvSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        # src = [src_len, bs]
        # trg = [trg_len, bs]
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)

        #calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        #encoder_conved is output from final encoder conv. block
        #encoder_combined is encoder_conved plus (elementwise) src embedding plus 
        #  positional embeddings 
        encoder_conved, encoder_combined = self.encoder(src)
        #encoder_conved = [bs, src_len, emb_dim]
        #encoder_combined = [bs, src_len, emb_dim]
        
        #calculate predictions of next words
        #output is a batch of predictions for each word in the trg sentence
        output = self.decoder(trg, encoder_conved, encoder_combined)
        #output = [bs, trg_len, emb_dim]
        
        return output



# ----- Attention models -----

class AttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        n_heads,
        n_layers,
        dropout,
        device,
        input_max_length=128,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = device
        self.input_max_length = input_max_length

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(input_max_length, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)

        encoder_layer = nn.TransformerEncoderLayer(emb_dim, n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src, src_mask):
        # src = [bs, src_len]
        # src_mask = [bs, src_len]
        bs, src_len = src.shape
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(bs, 1).to(self.device)
        # pos = [bs, src_len]
        
        src = self.dropout(self.tok_embedding(src) + self.pos_embedding(pos))
        # src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [bs, src_len, emb_dim]

        src = src.permute(1, 0, 2)

        enc_src = self.encoder(src, src_key_padding_mask=src_mask)
        enc_src = enc_src.permute(1, 0, 2)
        # enc_src = [bs, src_len, emb_dim]

        return enc_src
        

class AttentionDecoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        n_heads,
        n_layers,
        dropout,
        device,
        output_max_length=128,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = device
        self.output_max_length = output_max_length

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(output_max_length, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(device)

        decoder_layer = nn.TransformerDecoderLayer(emb_dim, n_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.fc_out = nn.Linear(emb_dim, output_dim)


    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [bs, trg_len]
        # enc_src = [bs, src_len, emb_dim]
        # trg_mask = [bs * n_heads, trg_len, src_len]
        # src_mask = [bs, src_len]
        bs, trg_len = trg.shape
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(bs, 1).to(self.device)
        # pos = [bs, trg_len]
        
        trg = self.dropout(self.tok_embedding(trg) + self.pos_embedding(pos))
        # trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [bs, trg_len, emb_dim]

        trg = trg.permute(1, 0, 2)
        enc_src = enc_src.permute(1, 0, 2)

        dec_trg = self.decoder(
            tgt=trg, 
            memory=enc_src, 
            tgt_mask=trg_mask, 
            memory_key_padding_mask=src_mask,
        )
        dec_trg = dec_trg.permute(1, 0, 2)
        # dec_trg = [bs, trg_len, emb_dim]

        output = self.fc_out(dec_trg)
        # output = [bs, trg_len, output_dim]

        return output


class AttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        device,
        src_pad_idx,
        trg_pad_idx,
        trg_init_idx=None,
        trg_eos_idx=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_init_idx = trg_init_idx
        self.trg_eos_idx = trg_eos_idx

    def make_src_mask(self, src):
        #src = [bs, src_len]

        src_mask = (src == self.src_pad_idx)
        #src_mask = [bs, 1, 1, src_len]

        return src_mask
    
    def make_trg_mask(self, trg):
        # trg = [bs, trg_len]
        trg_len = trg.shape[1]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [bs, 1, 1, trg_len]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg_len, trg_len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [bs, 1, trg_len, trg_len]
        
        trg_mask = trg_mask.squeeze(1).repeat(self.decoder.n_heads, 1, 1)
        # trg_mask = [bs * n_heads, trg_len, trg_len]
        # we need to prepare lower triangular mask matrix with such dimensions for torch.nn.MultiHeadAttention

        return ~trg_mask

    def forward(self, src, trg, tune_bert=False):
        # src = [src_len, bs]
        # trg = [trg_len, bs]

        src = src.permute(1, 0)
        trg = trg.permute(1, 0)

        bs, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # calculate encoded src
        if "AttentionEncoder" in str(type(self.encoder)): 
            enc_src = self.encoder(src, src_mask)
        else:
            enc_src = self.encoder(src, tune_bert)
        # enc_src = [bs, src_len, emb_dim]
        
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        output = output.permute(1, 0, 2)
        # output = [trg_len, bs, output_dim]

        return output

    # def translate(self, src, greedy=True, eps=1e-10):
    #     # src = [bs, src_len]

    #     src_mask = self.make_src_mask(src)
    #     enc_src = model.encoder(src, src_mask)

    #     bs = src.shape[0]
    #     trg_vocab_size = self.decoder.output_dim

    #     generated = torch.LongTensor([self.trg_init_idx]).repeat(bs, 1).to(self.device)
    #     logits_seq = [torch.log(to_one_hot(generated[:,0], trg_vocab_size) + eps)]

    #     for t in range(1, self.decoder.output_max_length):
    #         trg_mask = model.make_trg_mask(generated)
            
    #         logits = model.decoder(generated, enc_src, trg_mask, src_mask)[:, -1, :]
    #         # logits = [bs, output_dim]
    #         if greedy:
    #             gen_t = logits.argmax(dim=1)
    #         else:
    #             probs = F.softmax(logits, dim=1)
    #             gen_t = torch.multinomial(probs, 1)[:, 0]

    #         generated = torch.hstack([generated, gen_t.unsqueeze(1)])
    #         # gen_tensor = [bs, t+1]

    #         logits_seq.append(logits)

    #     # generated = [bs, output_dim], log_probs = [bs, max_len, output_dim]
    #     return generated, F.log_softmax(torch.stack(logits_seq, 1), dim=-1)


class BERTEncoder(nn.Module):
    def __init__(
        self,
        bert_encoder,
        emb_dim,
    ):
        super().__init__()

        self.encoder = bert_encoder
        self.bert_out_dim = self.encoder.pooler.dense.out_features
        self.emb_dim = emb_dim
        self.fc = nn.Linear(self.bert_out_dim, self.emb_dim)

    def forward(self, src, tune_bert=False):
        # src = [bs, src_len]
        bs, src_len = src.shape

        if tune_bert:
            enc_src = self.encoder(src)[0]
        else:
            with torch.no_grad():
                enc_src = self.encoder(src)[0]
        # enc_src = [bs, src_len, bert_out_dim]

        enc_src = self.fc(enc_src)
        # enc_src = [bs, src_len, emb_dim]
        return enc_src



# ----- Model utility functions -----

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
