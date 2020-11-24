#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from ast import literal_eval
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import time
import random


# In[2]:


url = "https://raw.githubusercontent.com/lmu-mandy/project-rgt/bob-branch/ted_talks_en.csv"
df = pd.read_csv(url)
df = df.loc[:, ['talk_id', 'topics', 'transcript']]
df.head()


# In[3]:


sep_topics = df.topics.unique()
topics = []

for topic in sep_topics:
    for i in topic.split(","):
        topics.append(i.split("'")[1])
print(topics[0:5])


# In[4]:


unique_topics = [] 
      
# traverse for all elements 
for topic in topics: 
    # check if exists in unique_list or not 
    if topic not in unique_topics: 
            unique_topics.append(topic) 
print(unique_topics)


# In[5]:


def find_topic(topic):
    """Returns a list of booleans for talks that contain a topic by index.
    
    :param topic: Topics or related topics of a talk
    """
    has_topic = []
    for t_list in df['topics']:
        if topic.lower() in literal_eval(t_list):
            has_topic.append(1)
        else:
            has_topic.append(0)
    return has_topic


# In[6]:


# add columns for selected topics
df['is_science'] = find_topic('science')
df['is_technology'] = find_topic('technology')
df['is_math'] = find_topic('math')
df['is_computers'] = find_topic('computers')
df['is_engineering'] = find_topic('engineering')
df['is_ML'] = find_topic('machine learning')
df['is_software'] = find_topic('software')
df['is_statistics'] = find_topic('statistics')
df['is_cognitive_science'] = find_topic('cognitive science')
df['is_science_and_art'] = find_topic('science and art')
df['is_physics'] = find_topic('physics')
df['is_quantum_physics'] = find_topic('quantum physics')
df['is_code'] = find_topic('code')
df['is_programming'] = find_topic('programming')
df['is_chemistry'] = find_topic('chemistry')
df['is_data'] = find_topic('data')
df.head()


# In[7]:


# filter DataFrame to only include talks about sex, religion, and politics
df = df.loc[(df['is_science']==1) | (df['is_technology']==1) | 
            (df['is_math']==1) | (df['is_computers']==1) |
            (df['is_engineering']==1) | (df['is_ML']==1) | 
            (df['is_software'] == 1) | (df['is_statistics'] == 1) | 
            (df['is_cognitive_science'] == 1) | (df['is_science_and_art'] == 1) | 
            (df['is_physics'] == 1) | (df['is_quantum_physics'] == 1) | 
            (df['is_code'] == 1) | (df['is_programming'] == 1) | 
            (df['is_chemistry'] == 1) | df['is_data'] == 1, : ].reset_index(drop=True)

# create new DataFrames for each topic (for later use)
science_df = df.loc[(df['is_science']==1), 'talk_id':'transcript'].reset_index(drop=True)
technology_df = df.loc[(df['is_technology']==1), 'talk_id':'transcript'].reset_index(drop=True)
math_df = df.loc[(df['is_math']==1), 'talk_id':'transcript'].reset_index(drop=True)
computers_df = df.loc[(df['is_computers']==1), 'talk_id':'transcript'].reset_index(drop=True)
engineering_df = df.loc[(df['is_engineering']==1), 'talk_id':'transcript'].reset_index(drop=True)
ML_df = df.loc[(df['is_ML']==1), 'talk_id':'transcript'].reset_index(drop=True)
software_df = df.loc[(df['is_software']==1), 'talk_id':'transcript'].reset_index(drop=True)
statistics_df = df.loc[(df['is_statistics']==1), 'talk_id':'transcript'].reset_index(drop=True)
cognitive_science_df = df.loc[(df['is_cognitive_science']==1), 'talk_id':'transcript'].reset_index(drop=True)
science_and_art_df = df.loc[(df['is_science_and_art']==1), 'talk_id':'transcript'].reset_index(drop=True)
physics_df = df.loc[(df['is_physics']==1), 'talk_id':'transcript'].reset_index(drop=True)
quantum_physics_df = df.loc[(df['is_quantum_physics']==1), 'talk_id':'transcript'].reset_index(drop=True)
code_df = df.loc[(df['is_code']==1), 'talk_id':'transcript'].reset_index(drop=True)
programming_df = df.loc[(df['is_programming']==1), 'talk_id':'transcript'].reset_index(drop=True)
chemistry_df = df.loc[(df['is_chemistry']==1), 'talk_id':'transcript'].reset_index(drop=True)
data_df = df.loc[(df['is_data']==1), 'talk_id':'transcript'].reset_index(drop=True)

print('Science', science_df.shape)
print('Technology', technology_df.shape)
print('Math', math_df.shape)
print('Computers', computers_df.shape)
print('Engineering', engineering_df.shape)
print('Machine Learning', ML_df.shape)
print('Software', software_df.shape)
print('Statistics', statistics_df.shape)
print('Cognitive Science', cognitive_science_df.shape)
print('Science and Art', science_and_art_df.shape)
print('Physics', physics_df.shape)
print('Quantum Physics', quantum_physics_df.shape)
print('Code', code_df.shape)
print('Programming', programming_df.shape)
print('Chemistry', chemistry_df.shape)
print('Data', data_df.shape)


# In[8]:


def combine_transcripts(transcript_list):
    """Input a list of transcripts and return them as a corpus.
    :param list_of_text: Transcript list"""
    corpus = ' '.join(transcript_list)
    return corpus

def transcripts_to_dict(df, topic_list):
    """Returns a dictionary of transcripts for each topic.
    
    :param df: DataFrame
    :param topic_list: List of topics
    """
    ted_dict = {}
    for topic in topic_list:
        # filter DataFrame to specific series and convert it to a list
        filter_string = 'is_' + str(topic)
        text_list = df.loc[(df[filter_string] == 1), 'transcript'].to_list()

        # call combine_transcripts function to return combined text
        combined_text = combine_transcripts(text_list)

        # add combined text to dict
        ted_dict[topic] = combined_text
    return ted_dict


# In[9]:


# create dictionary from the DataFrame
transcript_dict = transcripts_to_dict(df, ['science', 'technology', 'math', 'computers', 'engineering', 'ML', 
                                           'software', 'statistics', 'cognitive_science', 'science_and_art', 'physics', 
                                           'quantum_physics', 'code', 'programming', 'chemistry', 'data'])

# construct DataFrame from dictionary
df = pd.DataFrame.from_dict(transcript_dict, orient='index')
df.rename({0: 'transcript'}, axis=1, inplace=True)

df


# In[10]:


def clean_text(text):
    """Returns clean text.
    Removes:
        *text in square brackets & parenthesis
        *punctuation
        *words containing numbers
        *double-quotes, dashes
    """
#     text = text.lower()
    text = re.sub('[\[\(].*?[\)\]]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[\“\–]', '', text)
    return text


# In[11]:


# clean text
df['transcript'] = pd.DataFrame(df['transcript'].apply(lambda x: clean_text(x)))
science_df['transcript'] = pd.DataFrame(science_df['transcript'].apply(lambda x: clean_text(x)))
technology_df['transcript'] = pd.DataFrame(technology_df['transcript'].apply(lambda x: clean_text(x)))
math_df['transcript'] = pd.DataFrame(math_df['transcript'].apply(lambda x: clean_text(x)))
computers_df['transcript'] = pd.DataFrame(computers_df['transcript'].apply(lambda x: clean_text(x)))
engineering_df['transcript'] = pd.DataFrame(engineering_df['transcript'].apply(lambda x: clean_text(x)))
ML_df['transcript'] = pd.DataFrame(ML_df['transcript'].apply(lambda x: clean_text(x)))
software_df['transcript'] = pd.DataFrame(software_df['transcript'].apply(lambda x: clean_text(x)))
statistics_df['transcript'] = pd.DataFrame(statistics_df['transcript'].apply(lambda x: clean_text(x)))
cognitive_science_df['transcript'] = pd.DataFrame(cognitive_science_df['transcript'].apply(lambda x: clean_text(x)))
science_and_art_df['transcript'] = pd.DataFrame(science_and_art_df['transcript'].apply(lambda x: clean_text(x)))
physics_df['transcript'] = pd.DataFrame(physics_df['transcript'].apply(lambda x: clean_text(x)))
quantum_physics_df['transcript'] = pd.DataFrame(quantum_physics_df['transcript'].apply(lambda x: clean_text(x)))
code_df['transcript'] = pd.DataFrame(code_df['transcript'].apply(lambda x: clean_text(x)))
programming_df['transcript'] = pd.DataFrame(programming_df['transcript'].apply(lambda x: clean_text(x)))
chemistry_df['transcript'] = pd.DataFrame(chemistry_df['transcript'].apply(lambda x: clean_text(x)))
data_df['transcript'] = pd.DataFrame(data_df['transcript'].apply(lambda x: clean_text(x)))


# In[12]:


dfs = [science_df, technology_df, math_df, computers_df, engineering_df, ML_df,
       software_df, statistics_df, cognitive_science_df, science_and_art_df, physics_df, 
       quantum_physics_df, code_df, programming_df, chemistry_df, data_df]
       
comb_df = pd.concat(dfs)


# In[13]:


comb_df.drop_duplicates().reset_index(drop=True)


# In[14]:


#comb_df
scripts = comb_df["transcript"].to_numpy()


# In[15]:


# preprocesses scripts to change '.' , '!', '?' to '<eos>' and adds a space between words and ','
new_scripts = []
length_sents = []

for i in range(len(scripts)):
  sents = scripts[i].split('.')

  new_sents = []
  for sent in sents:
    sent = sent + ' ' + '<eos>'
    new_sents.append(sent)

  comb_sents = ''.join(new_sents).split('!')

  new_sents2 = []
  for sent in comb_sents:
    sent = sent + ' ' + '<eos>'
    new_sents2.append(sent)
  comb_sents2 = ''.join(new_sents2).split('?')

  new_sents3 = []
  for sent in comb_sents2:
    sent = sent + ' ' + '<eos>'
    new_sents3.append(sent)
  comb_sents3 = ''.join(new_sents3).split(',')

  new_sents4 = []
  for sent in comb_sents3:
    sent = sent + ' ' + ','
    new_sents4.append(sent)
  comb_sents4 = ''.join(new_sents4).split(';')

  new_sents5 = []
  for sent in comb_sents4:
    sent = sent + ' ' + ';'
    new_sents5.append(sent)
  comb_sents5 = ''.join(new_sents5)

  sen_lengths = []
  split_sents = comb_sents5.split('<eos>')

  for sent in split_sents:
    words = sent.split(' ')
    length = len(words)
    sen_lengths.append(length)

  length_sents.append(sen_lengths)
  new_scripts.append(comb_sents5)


# In[16]:


list_idx = []
for i in range(len(length_sents)):
  max_num = max(length_sents[i])
  if max_num >= 80:
    list_idx.append(i)

for i in range(len(list_idx)):
  new_scripts.pop(list_idx[i] - i)


# In[17]:


def load_vocab(text):
    word_to_ix = {}
    for sent in text:
        for word in sent.split():
            word = word.lower()
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix


# In[18]:


train_data1 = []
for idx in range(len(new_scripts)):
  split_sents = new_scripts[idx].split('<eos>')

  sentence = []
  for sent in split_sents:
    words = sent.split()
    words = [words[i].lower() for i in range(len(words))]
    words.append('<eos>')
    num = len(words)
    if len(words) <= 80:
      [words.append('<pad> ') for i in range(80 - num)]
    sentence.append(' '.join(words))
  data = ''.join(sentence)
  train_data1.append(data)

word2idx = load_vocab(train_data1)

train_data = []
for sent in train_data1:
    words = []
    for word in sent.split():
      words.append(word2idx[word])
    train_data.append(words)


# In[19]:


# separate by sentence
sent_length = 80
pad = word2idx['<pad>']
for idx in range(0, len(train_data)):
    seq_len = len(train_data[idx])
    if seq_len > sent_length: # if greater than sent_length, add the rest of current sequence at the end of transcript 
        iterations = math.ceil(seq_len / sent_length)
        for j in range(1, iterations): # first sent_length words are appended at end of for loop
            start_idx = sent_length * j
            end_idx = start_idx + sent_length
            new_data = train_data[idx][start_idx:end_idx] # add new data of sequence length at end of train_data
            new_seq_len = len(new_data)
            if new_seq_len < sent_length: # if sequence length not long enough, add padding (make zero for rest of length)
                [new_data.append(pad) for i in range(sent_length - new_seq_len)] # for padding
            train_data.append(new_data) 
    if seq_len < sent_length: # if sequence length not long enough, add padding (make zero for rest of length)
        print("true", idx)
        [train_data[idx].append(pad) for i in range(sent_length - seq_len)] # for padding
    train_data[idx] = train_data[idx][0:sent_length] # cut off at sequence length


# In[20]:


# check if worked
idx1 = 0
idx2 = 500
print(train_data[idx1])
print(len(train_data[idx1]))
print(train_data[idx2])
print(len(train_data[idx2]))
print(train_data[-1])
print(len(train_data[-1]))


# In[21]:


device = torch.device("cuda")

# 80 percent train, 10 percent validation, 10 percent test split

end1 = round(len(train_data)*.8) # to get 80% for training
end2 = round(len(train_data)*.9) # to get 10% for validation and test
print(end1)
print(end2)

random.shuffle(train_data) # shuffle the data to get a better distribution

train = torch.Tensor(train_data[0:end1])
val_data = torch.Tensor(train_data[end1:end2])
test_data = torch.Tensor(train_data[end2:])

train_data = train.long()
val_data = val_data.long()
test_data = test_data.long()


# In[22]:


class twoLayer_LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, layers):
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, hidden_size)
        self.rec_layer = nn.LSTM(hidden_size, hidden_size, num_layers=layers)
        self.lin_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_seq, h_init, c_init):
        g_seq = self.emb_layer(word_seq)  
        h_seq, (h_last, c_last) = self.rec_layer(g_seq, (h_init, c_init))
        score_seq = self.lin_layer(h_seq)
        return score_seq, (h_last, c_last)


# In[23]:


def evaluate(data):
    running_loss = 0
    num_batches = 0    
    with torch.no_grad():
        h = torch.zeros(layers, bs, hidden_size)
        c = torch.zeros(layers, bs, hidden_size)
        h = h.to(device)
        c = c.to(device)
        for count in range(0, len(data) - seq_length, seq_length):
            minibatch_data = data[count:count + seq_length]
            minibatch_label = data[count+1:count + seq_length + 1]
            minibatch_data = minibatch_data.to(device)
            minibatch_label = minibatch_label.to(device)

            scores, (h, c) = net(minibatch_data, h, c)

            minibatch_label = minibatch_label.view(bs * seq_length) 
            scores = scores.view(bs*seq_length, vocab_size)

            loss = criterion(scores, minibatch_label)    

            h = h.detach()
            c = c.detach()

            #running_loss += loss.item()
            num_batches += 1        
    
    #total_loss = running_loss/num_batches 
    print('test loss =', loss.item())


# In[24]:


# setup NN
hidden_size = 300
vocab_size = len(word2idx)
layers = 2
num_epoch = 10
bs = 80
seq_length = sent_length
my_lr = 0.9

net = twoLayer_LSTM(vocab_size, hidden_size, layers)
net.emb_layer.weight.data.uniform_(-0.1, 0.1)
net.lin_layer.weight = net.emb_layer.weight
net = net.to(device)
criterion = nn.CrossEntropyLoss()
train_size = len(train_data)
#test_size = len(test_data)
optimizer = optim.SGD(net.parameters(), lr=my_lr)


# In[25]:


def normalize_gradient(net):
    grad_norm_sq = 0
    for p in net.parameters():
        grad_norm_sq += p.grad.data.norm()**2
    grad_norm = math.sqrt(grad_norm_sq)
    if grad_norm < 1e-4:
        net.zero_grad()
        print('grad norm close to zero')
    else:    
        for p in net.parameters():
             p.grad.data.div_(grad_norm)
    return grad_norm


# In[26]:


# training
start = time.time()
n = 0.5
for epoch in range(num_epoch):
    if epoch >= 4: 
        if my_lr > 0.06:
            my_lr = my_lr/(4*n)
        n += 0.5
        optimizer = optim.SGD(net.parameters(), lr=my_lr) 
            
    # set the running quantities to zero at the beginning of the epoch
    running_loss = 0
    num_batches = 0    
       
    # set the initial h to be the zero vector
    h = torch.zeros(layers, bs, hidden_size)
    c = torch.zeros(layers, bs, hidden_size)
    # send it to the gpu    
    h = h.to(device)
    c = c.to(device)

    for count in range(0, train_size - seq_length, seq_length):    
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # create a minibatch
        minibatch_data = train_data[count:count + seq_length]
        minibatch_label = train_data[count + 1:count + seq_length+1]        
                
        # send them to the gpu
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)
        
        # Detach to prevent from backpropagating all the way to the beginning
        # Then tell Pytorch to start tracking all operations that will be done on h and c
        h = h.detach()
        c = c.detach()
        h = h.requires_grad_()
        c = c.requires_grad_()
        # forward the minibatch through the net 
        scores, (h, c) = net(minibatch_data, h, c)
        # reshape the scores and labels to huge batch of size bs*seq_length
        scores = scores.view(bs * seq_length, vocab_size)  
        minibatch_label = minibatch_label.view(bs * seq_length)       
        
        # Compute the average of the losses of the data points in this huge batch
        loss = criterion(scores, minibatch_label)
        
        # backward pass to compute dL/dR, dL/dV and dL/dW
        loss.backward()

        # do one step of stochastic gradient descent: R=R-lr(dL/dR), V=V-lr(dL/dV), ...
        normalize_gradient(net)
        optimizer.step()
        
        # update the running loss  
        #running_loss += loss.item()
        num_batches += 1
                          
    #total_loss = running_loss/num_batches
    elapsed = time.time() - start
    print('\nepoch =', epoch, '\t time =', elapsed,'\t lr =', my_lr, '\t training loss =', loss.item()) # compute error on the test set at end of each epoch
    evaluate(val_data) # eval on the validation set

print(" ")


# In[27]:


param = net.state_dict()
torch.save(param, 'trained_parameters_LSTM.pt')


# In[28]:


idx2word = {y:x for x, y in word2idx.items()}


# In[29]:


def show_most_likely_words(prob):
    num_word_display = 15
    p = prob.view(-1)
    p, word_idx = torch.topk(p, num_word_display)
    for i, idx in enumerate(word_idx):
        percentage = p[i].item() * 100
        word = idx2word[idx.item()]
        print("{:.1f}%\t".format(percentage), word) 

def text2tensor(text):
    text = text.lower()
    list_of_words = text.split()
    list_of_int = [word2idx[w] for w in list_of_words]
    x = torch.LongTensor(list_of_int)
    return x


# In[36]:


sentence = "machine learning is weird and "
bs=20
h = torch.zeros(layers, bs, hidden_size)
c = torch.zeros(layers, bs, hidden_size)
h = h.to(device)
c = c.to(device)

data = text2tensor(sentence)
seqLength = len(data)
data = data.view(seqLength,-1)
empty = torch.zeros(seqLength,19).type(torch.LongTensor)
data = torch.cat((data,empty),dim=1)
data = data.to(device)
scores, (h,c)  = net(data,h,c)
scores = scores[seq_length-1, 0, :]
p = F.softmax(scores.view(1, vocab_size), dim=1)
print(sentence, '... \n')
show_most_likely_words(p)


# In[ ]:





# In[37]:


torch.save(net, 'entire_model.pth')


# In[38]:


model_new = torch.load('entire_model.pth')


# In[39]:


data = text2tensor(sentence)
seqLength = len(data)
data = data.view(seqLength,-1)
empty = torch.zeros(seqLength,19).type(torch.LongTensor)
data = torch.cat((data,empty),dim=1)
data = data.to(device)
scores, (h,c)  = model_new(data,h,c)
scores = scores[seq_length-1, 0, :]
p = F.softmax(scores.view(1, vocab_size), dim=1)
print(sentence, '... \n')
show_most_likely_words(p)


# In[ ]:




