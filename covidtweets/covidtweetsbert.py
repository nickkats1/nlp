import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertModel,BertTokenizer,AdamW,get_linear_schedule_with_warmup
import numpy as np
import warnings
import re
import demoji
import pandas as pd
from textblob import TextBlob
import contractions
import unicodedata

warnings.filterwarnings("ignore")

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv("C:/nlp_projects/covidtweets/vaccination_all_tweets.csv",delimiter=',',nrows=10000)
df = df[['text']]
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)


def clean_text(text):

    text = text.lower()


    text = re.sub(r'<.*?>', '', text)


    text = contractions.fix(text)

    text = re.sub(r'https?://\S+|www\.\S+', '', text)


    text = re.sub(r'@\w+|#\w+', '', text)


    text = demoji.replace(text, '')

    text = unicodedata.normalize("NFKD", text)


    text = re.sub(r'[^a-zA-Z\s]', '', text)

    

    return text





df['text'] = df['text'].apply(clean_text)
df['text'].head(10)



def polarity(text):
    return TextBlob(text).polarity

df['polarity'] = df['text'].apply(polarity)


def sentiment(label):
    if label == 0:
        return "NEUTRAL"
    elif label < 0:
        return "NEGATIVE"
    elif label > 0:
        return "POSITIVE"



df['sentiment'] = df['polarity'].apply(sentiment)

plt.figure(figsize=(10,6))
df['sentiment'].value_counts().plot(kind='bar',rot=0)
plt.show()


df['sentiment'] = df['sentiment'].map({"NEGATIVE":0,"NEUTRAL":1,"POSITIVE":2})


df.dtypes

df = df[['text','sentiment']]
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.dtypes

MODEL_NAME = "bert-base-cased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


sample_tokens = []
for txt in df['text']:
    tokens = tokenizer.encode(txt,max_length=512)
    sample_tokens.append(len(tokens))


plt.figure(figsize=(10,6))
sns.distplot(sample_tokens)
plt.show()



MAX_LEN = 50


class custom_dataset(Dataset):
    def __init__(self,text,targets,max_len,tokenizer):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,idx):
        text = str(self.text[idx])
        target = self.targets[idx]
        
        encoder = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt',
            )
        
        return {
            "text":text,
            "input_ids":encoder['input_ids'].flatten(),
            "attention_mask":encoder['attention_mask'].flatten(),
            "targets":torch.tensor(target,dtype=torch.long)
            }




from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size=.20,random_state=42)
df_val,df_test = train_test_split(df_test, test_size=.50,random_state=42)


BATCH_SIZE = 14

def get_dataloader(df,tokenizer,batch_size,max_len):
    ds = custom_dataset(
        text = df['text'].to_numpy(),
        targets = df['sentiment'].to_numpy(),
        max_len=max_len,
        tokenizer=tokenizer
        )
    return DataLoader(
        ds,
        num_workers=0,
        batch_size=batch_size
        )

train_dataloader = get_dataloader(df_train, tokenizer, BATCH_SIZE, MAX_LEN)
test_dataloader = get_dataloader(df_test,tokenizer,BATCH_SIZE,MAX_LEN)
val_dataloader = get_dataloader(df_val,tokenizer,BATCH_SIZE,MAX_LEN)




class Bert_Classifier(nn.Module):
    def __init__(self, n_classes=3):
        super(Bert_Classifier,self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(768,3)
    

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)



model = Bert_Classifier(3)
model.to(device)


num_epochs = 4
total_steps = len(train_dataloader) * num_epochs


optimizer = AdamW(model.parameters(),lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps)




def training_epoch(model,dataloader,loss_fn,optimizer,device,scheduler,n_examples):
    model.train()
    losses = []
    predictions = 0
    for d in dataloader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        
        outputs = model(
            attention_mask=attention_mask,
            input_ids=input_ids
            )
        _,preds = torch.max(outputs,dim=1)
        loss = loss_fn(outputs,targets)
        predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return (predictions.double() / n_examples,np.mean(losses))



def eval_model(model,dataloader,loss_fn,device,n_examples):
    model.eval()
    losses = []
    predictions = 0
    
    
    with torch.no_grad():
        for d in dataloader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(
                attention_mask=attention_mask,
                input_ids=input_ids
                )
            
            _,preds = torch.max(outputs,dim=1)
            loss = loss_fn(outputs,targets)
            
            predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
        return (predictions.double() / n_examples,np.mean(losses))






for epoch in range(num_epochs):
    print(f'{epoch+1}/{num_epochs}')
    
    train_acc,train_loss = training_epoch(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
        )
    
    
    val_acc,val_loss = eval_model(model, val_dataloader, loss_fn, device, len(df_val))
    print(f" Training accuracy {train_acc*100:.2f}%; -Val Accuracy-- {val_acc*100:.2f}%")
    print(f"training loss {train_loss*100:.2f}%; --val loss-- {val_loss*100:.2f}%")





test_acc, _ = eval_model(
  model,
  test_dataloader,
  loss_fn,
  device,
  len(df_test)
)

print(f'Testing Accuracy : {test_acc.item() *100 :.2f}%')







from sklearn.metrics import classification_report,confusion_matrix


def get_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for d in dataloader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)
            
            outputs = model(
                attention_mask=attention_mask,
                input_ids=input_ids
            )
            
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    return all_labels, all_preds





true_labels, predicted_labels = get_predictions(model, test_dataloader, device)

clf_rpt = classification_report(true_labels,predicted_labels)
print(f'Classification Report: {clf_rpt}')




def plot_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",xticklabels=['NEGATIVE','NEUTRAL','POSITIVE'],yticklabels=['NEGATIVE','NEUTRAL','POSITIVE'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


plot_confusion_matrix(true_labels, predicted_labels)




