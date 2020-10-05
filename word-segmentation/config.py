model_name = "BiGRU" # BiGRU or BiLSTM
dataset = "msr" # msr or pku
embedding_dim = 50
drop_rate = 0.2 
rnn_dim=32

# compile
learning_rate=0.001
beta_1=0.9
beta_2=0.999

# fit
batch_size = 20
epochs = 10
workers = 8

# dev
is_test_mode = False