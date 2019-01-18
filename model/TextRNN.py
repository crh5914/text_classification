import tensorflow as tf
from tensorflow.contrib import rnn

class TextRNN:
    def __init__(self,num_class,vocab_size,emb_dim,hidden_dim,lr,dropout_keep_prob=None,learner='adam'):
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_keep_prob=dropout_keep_prob
        #self.seq_length = seq_length
        self.lr = lr
        self.learner = learner
        self.build_up()
    def build_up(self):
        self.x = tf.placeholder(shape=(None,None),dtype=tf.int32)
        self.y = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.word_table = tf.Variable(tf.truncated_normal(shape=(self.vocab_size,self.emb_dim),stddev=0.1),name="word_table")
        word_emb = tf.nn.embedding_lookup(self.word_table,self.x)
        fw_cell = rnn.BasicLSTMCell(self.hidden_dim)
        bw_cell = rnn.BasicLSTMCell(self.hidden_dim)
        if self.dropout_keep_prob is not None:
            fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=self.dropout_keep_prob)
            bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=self.dropout_keep_prob)
        outputs,_ =tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,word_emb,dtype=tf.float32)
        output_rnn = tf.concat(outputs,axis=2) #(None,seq_length,hidden_dim*2)
        output_rnn_last = output_rnn[:,-1,:] #(None,hidden_dim*2)
        W1 = tf.Variable(tf.truncated_normal(shape=(self.hidden_dim*2,self.hidden_dim),stddev=0.1),name="W1")
        b1 = tf.Variable(tf.constant(0.0,shape=(self.hidden_dim,)),name="b1")
        W2 = tf.Variable(tf.truncated_normal(shape=(self.hidden_dim,self.num_class),stddev=0.1),name="W2")
        b2 = tf.Variable(tf.constant(0.0,shape=(self.num_class,)),name="b2")
        f1 = tf.nn.relu(tf.matmul(output_rnn_last,W1) + b1)
        f2 = tf.nn.relu(tf.matmul(f1,W2) + b2)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=f2,labels=self.y)
        y_ = tf.nn.softmax(f2)
        self.y_pred = tf.argmax(y_,axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y_pred,self.y),tf.float32))
        if self.learner is "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.learner is 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_step = optimizer.minimize(self.loss)





