from transformers import BertModel,BertTokenizer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import re
import demoji
import string
import pandas as pd
import tqdm.notebook as tq
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,f1_score

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)


df = pd.read_csv("C:/mhbert/Combined Data.csv",delimiter=',',nrows=10000)
df.head(10)

df.drop("Unnamed: 0",inplace=True,axis=1)
df.isnull().sum()
df['statement'] = df['statement'].fillna("")

df.duplicated().sum()
df.drop_duplicates(inplace=True)


plt.figure(figsize=(10,6))
df['status'].value_counts().plot(kind='bar',rot=0)
plt.show()



df['Normal'] = (df['status'] == "Normal").astype(int)
df['Depression'] = (df['status'] == 'Depression').astype(int)
df['Suicidal'] = (df['status'] == 'Suicidal').astype(int)
df['Anxiety'] = (df['status'] == 'Anxiety').astype(int)

df['status'].value_counts()


df.drop("status",inplace=True,axis=1)

TARGET_LIST = list(df.columns[1:])
print(TARGET_LIST)
df[TARGET_LIST]



df.dtypes

def clean_text(text):
    
    text = str(text).lower()

    text = re.sub(r'<.*?>', '',text)
    
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    text = demoji.replace(text,'')
    
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    
    return text



df['statement'] = df['statement'].apply(clean_text)
df['statement'].head(10)





MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME,do_lower_case=True)


sample_tokens = []
for txt in df['statement']:
    tokens = tokenizer.encode(txt, max_length=512)
    sample_tokens.append(len(tokens))


plt.figure(figsize=(10,6))
sns.distplot(sample_tokens)
plt.show()



MAX_LEN = 200


class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self,df,max_len,tokenizer,TARGET_LIST):
        self.df = df
        self.statement = list(df['statement'])
        self.targets = self.df[TARGET_LIST].values
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.statement)
    
    def __getitem__(self,idx):
        statement = str(self.statement[idx])
        statement = " ".join(statement.split())
        
        encoder = self.tokenizer.encode_plus(
            statement,
            None,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt',
            )
        
        return {
            "statement":statement,
            "input_ids":encoder['input_ids'].flatten(),
            "attention_mask":encoder['attention_mask'].flatten(),
            "token_type_ids":encoder['token_type_ids'].flatten(),
            "targets":torch.FloatTensor(self.targets[idx])
            }


from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size=.20,random_state=42)
df_val,df_test = train_test_split(df_test, test_size=.50,random_state=42)







train_dataset = Custom_Dataset(df_train,MAX_LEN,tokenizer,TARGET_LIST)
test_dataset = Custom_Dataset(df_test, MAX_LEN, tokenizer, TARGET_LIST)
val_dataset = Custom_Dataset(df_test, MAX_LEN, tokenizer, TARGET_LIST)


BATCH_SIZE = 8


train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,num_workers=0)


data= next(iter(train_dataloader))
data.keys()

data['targets'].shape
data['input_ids'].shape

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained(MODEL_NAME, return_dict=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(
            input_ids=input_ids, 
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



optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5)

EPOCHS = 4


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

    print(f'train_loss={train_loss*100:.2f}, val_loss={val_loss*100:.2f} train_acc={train_acc*100:.2f}, val_acc={val_acc*100:.2f}')




test_acc,test_loss = eval_model(test_dataloader, model, optimizer)
print(f'testing accuracy: {test_acc*100:.2f}%')
print(f'testing loss: {test_loss*100:.2f}')



def get_predictions(model, dataloader):
    model.eval()
    
    statements = []
    predictions = []
    prediction_probs = []
    target_values = []

    with torch.no_grad():
      for data in dataloader:
        statement = data["statement"]
        input_ids = data["input_ids"].to(device, dtype = torch.long)
        attention_mask = data["attention_mask"].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data["targets"].to(device, dtype = torch.float)

        outputs = model(input_ids, attention_mask, token_type_ids)
        _, preds = torch.max(outputs, dim=1)
        _, targ = torch.max(targets, dim=1)

        statements.extend(statement)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        target_values.extend(targ)
        
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    target_values = torch.stack(target_values).cpu()
    
    return statements, predictions, prediction_probs, target_values




statements, predictions, prediction_probs, target_values = get_predictions(model, test_dataloader)


print(classification_report(predictions,target_values, target_names=TARGET_LIST))

print(f1_score(predictions, target_values,average='micro'))

def plt_confusion_matrix(predictions,target_values):
    cm = confusion_matrix(predictions,target_values)
    sns.heatmap(cm, fmt='d',annot=True,cmap="Blues",xticklabels=["Normal","Depression","Suicidal","Anxiety"],yticklabels=["Normal","Depression","Suicidal","Anxiety"])



plt_confusion_matrix(predictions, target_values)

