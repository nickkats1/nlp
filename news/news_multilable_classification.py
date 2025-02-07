import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import numpy as np
import tqdm.notebook as tq
from sklearn.metrics import confusion_matrix,classification_report,f1_score


device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv("C:/mlc/bbc-text-1.csv",delimiter=',',nrows=20000)

df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df['category'].head(10)
df['category'].value_counts()
df['Text'] = df['text']
df.drop('text',inplace=True,axis=1)

df['sports'] = (df['category'] == "sport").astype(int)
df['business'] = (df['category'] == "business").astype(int)
df['politics'] = (df['category'] == "politics").astype(int)
df['entertainment'] = (df['category'] == "entertainment").astype(int)
df['tech'] = (df['category'] == "tech").astype(int)

df.drop("category",inplace=True,axis=1)


learning_rate = 1e-5
MAX_LEN = 256
EPOCHS = 4

MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)




class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self,df,labels,max_len,tokenizer):
        self.df = df
        self.Text = list(df['Text'])
        self.targets = self.df[labels].values
        self.max_len = max_len
        self.tokenizer=  tokenizer
        
        
    def __len__(self):
        return len(self.Text)
    
    
    def __getitem__(self, idx):
        Text = str(self.Text[idx])
        Text = " ".join(Text.split())
        
        encoder = self.tokenizer.encode_plus(
            Text,
            None,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
            return_tensors='pt',
            )
        
        return {
            "Text":Text,
            "input_ids":encoder['input_ids'].flatten(),
            "attention_mask":encoder['attention_mask'].flatten(),
            "token_type_ids":encoder['token_type_ids'].flatten(),
            "targets":torch.FloatTensor(self.targets[idx])
            }


from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size=.20,random_state=42)
df_val,df_test = train_test_split(df_test,test_size=.50,random_state=42)


labels = list(df.columns)
labels = labels[1:]

train_dataset = Custom_Dataset(df_train,labels,max_len=MAX_LEN,tokenizer=tokenizer)
test_dataset = Custom_Dataset(df_test, labels, max_len=MAX_LEN, tokenizer=tokenizer)
val_dataset = Custom_Dataset(df_val, labels, max_len=MAX_LEN, tokenizer=tokenizer)

TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4


train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,num_workers=0)
test_dataloader=  torch.utils.data.DataLoader(test_dataset,batch_size=TEST_BATCH_SIZE,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=VAL_BATCH_SIZE,num_workers=0)

data = next(iter(train_dataloader))
data.keys()





class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(MODEL_NAME, return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 5)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


    

model = BertClassifier()
model.to(device)



def loss_fn(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)



optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)



def train_model(train_dataloader, model, optimizer):
    model.train()
    losses = []
    correct_predictions = 0
    num_samples = 0
    model.train()
    loop = tq.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
    for batch_idx, data in loop:
        input_ids = data['input_ids'].to(device, dtype = torch.long)
        attention_mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)


        outputs = model(input_ids, attention_mask,token_type_ids)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        _, preds = torch.max(outputs, dim=1)
        _, targ = torch.max(targets, dim=1)
        num_samples += len(targ)
        correct_predictions += torch.sum(preds == targ)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


    return model, float(correct_predictions)/num_samples, np.mean(losses)



def eval_model(val_dataloader, model, optimizer):
    model.eval()
    losses = []
    correct_predictions = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader, 0):
            input_ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(input_ids, attention_mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())


            _, preds = torch.max(outputs, dim=1)
            _, targ = torch.max(targets, dim=1)
            num_samples += len(targ)
            correct_predictions += torch.sum(preds == targ)

    return float(correct_predictions)/num_samples, np.mean(losses)






for epoch in range(1, EPOCHS+1):
    print(f'Epoch {epoch}/{EPOCHS}')
    model, train_acc, train_loss = train_model(train_dataloader, model, optimizer)
    val_acc, val_loss = eval_model(val_dataloader, model, optimizer)

    print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.2f} train_acc={train_acc:.2f}, val_acc={val_acc:.4f}')




for epoch in range(1, EPOCHS+1):
    print(f'Epoch {epoch}/{EPOCHS}')
    model, train_acc, train_loss = train_model(train_dataloader, model, optimizer)
    val_acc, val_loss = eval_model(val_dataloader, model, optimizer)

    print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.2f} train_acc={train_acc:.2f}, val_acc={val_acc:.4f}')





def get_predictions(model, dataloader):
    model.eval()
    
    Texts = []
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
      for data in dataloader:
        Text = data["Text"]
        input_ids = data["input_ids"].to(device, dtype = torch.long)
        attention_mask = data["attention_mask"].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data["targets"].to(device, dtype = torch.float)

        outputs = model(input_ids, attention_mask, token_type_ids)
        _, preds = torch.max(outputs, dim=1)
        _, targ = torch.max(targets, dim=1)

        Texts.extend(Text)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targ)
        
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    target_values = torch.stack(target_values).cpu()
    
    return Texts, predictions, prediction_probs, target_values
     


Texts, predictions, prediction_probs, target_values = get_predictions(model, test_dataloader)


print(classification_report(predictions,target_values, target_names=labels))

print(f1_score(predictions, target_values,average='micro'))

def plt_confusion_matrix(predictions,target_values):
    cm = confusion_matrix(predictions,target_values)
    sns.heatmap(cm, fmt='d',annot=True,cmap="Blues",xticklabels=["sports","business","politics","entertainment","tech"],yticklabels=["sports","business","politics","entertainment","tech"])



plt_confusion_matrix(predictions, target_values)