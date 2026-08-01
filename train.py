import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AlphaZeroDataset(Dataset):

    def __init__(self, example):

        self.example = example

    def __len__(self):

        return len(self.example)

    def __getitem__(self, idx):
        state, prob, reward = self.example[idx]

        return(
            torch.FloatTensor(np.asarray(state, dtype = np.float32)).unsqueeze(0),
            torch.FloatTensor(prob),
            torch.FloatTensor([reward])
        )

class Trainer:

    def __init__(self, model, lr=0.001, batch_size = 64, epochs = 10):

        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train(self, examples):

        dataset = AlphaZeroDataset(examples)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        self.model.train()

        print(f"Starting training using {len(examples)} samples [{self.device}]")

        for epoch in range(self.epochs):

            total_loss = 0
            total_pi_loss = 0
            total_v_loss = 0

            for batch_states, batch_probs, batch_rewards in dataloader:

                batch_states = batch_states.to(self.device)
                batch_probs = batch_probs.to(self.device)
                batch_rewards = batch_rewards.to(self.device)

                self.optimizer.zero_grad()

                out_prob, out_v = self.model(batch_states)

                prob_loss = -(
                    batch_probs * F.log_softmax(out_prob, dim = 1)
                ).sum(dim = 1).mean()

                v_loss = F.mse_loss(out_v.squeeze(-1), batch_rewards.squeeze(-1))


                loss = prob_loss + v_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_pi_loss += prob_loss.item()
                total_v_loss += v_loss.item()

            avg_loss = total_loss / len(dataloader)
            avg_pi_loss = total_pi_loss / len(dataloader)
            avg_v_loss = total_v_loss / len(dataloader)

            print(f"Epoch: {epoch+1} of {self.epochs} | "
                  f"Loss: {avg_loss:.4f} | Policy: {avg_pi_loss:.4f} | Value: {avg_v_loss:.4f})")