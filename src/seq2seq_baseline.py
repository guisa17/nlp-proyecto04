import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.rnn import pad_sequence


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)


    def forward(self, src):
        emb = self.embedding(src)
        output, hidden = self.rnn(emb)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden):
        emb = self.embedding(input_token).unsqueeze(1)
        output, hidden = self.rnn(emb, hidden)
        pred = self.fc_out(output.squeeze(1))
        return pred, hidden
    

class Seq2SeqBaseline:
    def __init__(self, tokenizer, emb_size=32, hidden_size=64, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer

        vs = tokenizer.vocab_size
        self.encoder = EncoderRNN(vs, emb_size, hidden_size).to(self.device)
        self.decoder = DecoderRNN(vs, emb_size, hidden_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )


    def train(self, train_data, epochs=5, batch_size=32):
        self.encoder.train(); self.decoder.train()
        for ep in range(1, epochs+1):
            random.shuffle(train_data)
            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                # Tokenizar y padding
                src_seqs = [torch.tensor(self.tokenizer.encode(inp),
                                         dtype=torch.long)
                            for inp,_ in batch]
                tgt_seqs = [torch.tensor(self.tokenizer.encode(tgt),
                                         dtype=torch.long)
                            for _,tgt in batch]

                src_batch = pad_sequence(src_seqs, batch_first=True,
                                         padding_value=self.tokenizer.pad_token_id
                                        ).to(self.device)
                tgt_batch = pad_sequence(tgt_seqs, batch_first=True,
                                         padding_value=self.tokenizer.pad_token_id
                                        ).to(self.device)

                # Preparar inputs/targets del decoder
                dec_input  = tgt_batch[:, :-1]   # sin <eos>
                dec_target = tgt_batch[:, 1:]    # sin <sos>

                self.optimizer.zero_grad()
                _, hidden = self.encoder(src_batch)

                loss = 0
                token = dec_input[:, 0]
                
                # Teacher forcing
                for t in range(dec_target.size(1)):
                    pred, hidden = self.decoder(token, hidden)
                    loss += self.criterion(pred, dec_target[:, t])
                    token = dec_input[:, t]

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() / dec_target.size(1)

            avg_loss = total_loss / (len(train_data) / batch_size)
            print(f"Epoch {ep}/{epochs}  Loss: {avg_loss:.4f}")


    def evaluate(self, val_data):
        self.encoder.eval(); self.decoder.eval()
        correct = total = 0
        with torch.no_grad():
            for inp, tgt in val_data:
                seq = torch.tensor(self.tokenizer.encode(inp),
                                   dtype=torch.long).unsqueeze(0).to(self.device)
                _, hidden = self.encoder(seq)
                token = torch.tensor([self.tokenizer.sos_token_id],
                                     dtype=torch.long).to(self.device)
                pred_ids = []
                # Generar hasta <eos> o max 50 pasos
                for _ in range(50):
                    pred, hidden = self.decoder(token, hidden)
                    token = pred.argmax(1)
                    if token.item() == self.tokenizer.eos_token_id:
                        break
                    pred_ids.append(token.item())

                pred_str = self.tokenizer.decode(pred_ids)
                if pred_str == tgt:
                    correct += 1
                total += 1

        acc = correct / total if total>0 else 0
        print(f"Validation accuracy: {acc:.4f}")
