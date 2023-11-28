import time
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from resnet18 import Model
from data_prep import train_dataloader, valid_dataloader
from cfg import config

class Trainer:
    def __init__(self, config):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = Model(num_classes = config["model"]["num_classes"])
        self.epochs = config["train"]["epochs"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.set_optimizer(lr = float(config["train"]["lr"]), 
                                            weight_decay = float(config["train"]["weight_decay"]))

        self.train_losses = []
        self.val_losses = [] 
        self.f1_scores = []
        self.best_f1 = 0

    def train(self, device = 'cpu', plot = True):
        self.model.to(device)
        start = time.time()
        print(f'Training for {self.epochs} epochs on {device}')
        for epoch in range(1,self.epochs+1):
            print(f"Epoch {epoch}/{self.epochs}")
            self.model.train()
            train_loss = torch.tensor(0., device=device) 
            train_bar = tqdm(train_dataloader, total=len(train_dataloader))
            for step, batch in enumerate(train_bar):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                preds = self.model(images)
                loss = self.criterion(preds, labels)
                train_bar.set_description("epoch {} loss {}".format(epoch,loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    train_loss += loss * train_dataloader.batch_size
            
            if self.valid_dataloader is not None:
                y_trues = []
                y_preds = []
                self.model.eval() 
                valid_loss = torch.tensor(0., device=device)
                val_bar = tqdm(valid_dataloader,total=len(valid_dataloader))
                for step, batch in enumerate(val_bar):
                    with torch.no_grad():
                        images = batch[0].to(device)
                        labels = batch[1].to(device)
                        preds = self.model(images)
                        loss = self.criterion(preds, labels)
                        
                        valid_loss += loss * valid_dataloader.batch_size
                        y_preds.append(torch.argmax(preds, dim=1).cpu().numpy())
                        y_trues.append(labels.cpu().numpy())
                        
                y_trues = np.concatenate(y_trues,0)
                y_preds = np.concatenate(y_preds,0)
                train_loss = train_loss/len(train_dataloader.dataset)
                valid_loss = valid_loss/len(valid_dataloader.dataset)
                
                precision = precision_score(y_trues, y_preds, average = "weighted")
                recall = recall_score(y_trues, y_preds, average = "weighted")
                f1 = f1_score(y_trues, y_preds, average = "weighted")
                result = {
                "eval_recall": float(recall),
                "eval_precision": float(precision),
                "eval_f1": float(f1),
                "eval_loss": float(valid_loss)
                }
                
                self.train_losses.append(float(train_loss))
                self.val_losses.append(float(valid_loss))
                self.f1_scores.append(float(f1))
            
            if valid_dataloader is not None:
                print("***** Eval results *****")
                for key in  sorted(result.keys()):
                    print(key, str(round(result[key],4)))
                if result['eval_f1']>best_f1:
                    best_f1=result['eval_f1']
                    print("  "+"*"*20)
                    print("  Best f1:",round(best_f1,4))
                    print("  "+"*"*20)
                    torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, './model_best_f1.pth.tar')
                    print(f"model saved: {os.getcwd()}/model_best_f1.pth.tar")
                    
        end = time.time()
        print(f'Total training time: {end-start:.1f} seconds')
        if plot:
            self.plot_results()

    def set_optimizer(self, lr, weight_decay):
        params_1x = [param for name, param in self.model.named_parameters() if 'fc' not in str(name)]
        optimizer = torch.optim.Adam([{'params':params_1x}, 
                                      {'params': self.model.resnet.fc.parameters(), 'lr': lr*100}], 
                                      lr=lr, weight_decay=weight_decay)
        return optimizer
    
    def plot_results(self):
        fig, axs = plt.subplots(1, 2, figsize=(20, 6)) 
        axs[0].plot(range(len(self.train_losses)), self.train_losses, color='orange', label='train', linestyle='--')
        axs[0].plot(range(len(self.val_losses)), self.val_losses, color='blue', marker='o', label='val')
        axs[0].set_title('train/val loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Batch loss')
        axs[0].legend()
        axs[1].plot(range(len(self.f1_scores)), self.f1_scores, color='red', marker='o')
        axs[1].set_title('val F1-score')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('F1')
        plt.show()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(config)
    trainer.train(device = device, plot = True)