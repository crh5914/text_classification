from util import Corpus
from TextRNN import TextRNN
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from sklearn.utils import shuffle
import tensorflow  as tf
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Run Train Script")
    parser.add_argument('--num_class',type=int,default=2)
    parser.add_argument('--emb_dim',type=int,default=100)
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--hidden_dim',type=int,default=100)
    parser.add_argument('--file',type=str,default='../data/train.csv')
    parser.add_argument('--test_rate',type=float,default=0.2)
    parser.add_argument('--lr',type=float,default=0.001)
    return parser.parse_args()
def reader(file):
    df = pd.read_csv(file)
    docs,labels = df['question_text'].values,df['target'].values
    return docs,labels
def train(sess,model,train_x,train_y):
    feed_dict = {model.x:train_x,model.y:train_y}
    _,loss,acc = sess.run([model.train_step,model.loss,model.acc],feed_dict=feed_dict)
    return loss,acc
def test(sess,model,test_x,test_y):
    feed_dict = {model.x: test_x, model.y: test_y}
    y_pred = sess.run([model.y_pred], feed_dict=feed_dict)
    return y_pred

def generate_train_batch(x,y,batch_size=256):
    x,y = shuffle(x,y)
    num_batch = int(len(x) / batch_size)
    for i in range(num_batch):
        start = i*batch_size
        end = (i+1)*batch_size
        if end > len(x):
            end = len(x)
        yield  x[start:end],y[start:end]
def generate_test_batch(x,y,batch_size=256):
    x,y = shuffle(x,y)
    num_batch = int(len(x) / batch_size)
    for i in range(num_batch):
        start = i*batch_size
        end = (i+1)*batch_size
        if end > len(x):
            end = len(x)
        yield  x[start:end],y[start:end]
def run(model,train_x,test_x,train_y,test_y,epoch):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for idx in range(epoch):
        acc,loss,s = 0,0,0
        for batch_x,batch_y in generate_train_batch(train_x,train_y):
            bloss,bacc = train(sess,model,batch_x,batch_y)
            loss += bloss*len(batch_y)
            acc += bacc*len(batch_y)
            s += len(batch_y)
        loss,acc = loss/s,acc/s
        print("Train,loss:{},accuracy:{}".format(loss,acc))
        ys_,ys = [],[]
        for batch_x,batch_y in generate_test_batch(test_x,test_y):
            y_ = test(sess,model,batch_x,batch_y)
            ys_.extend(y_)
            ys.extend(batch_y)
        acc = accuracy_score(ys,ys_)
        f1 = f1_score(ys,ys_)
        print("Test,accuracy:{},f1-score:{}".format(acc,f1))
def main():
    args = parse_args()
    ds =Corpus(args.file,reader)
    vocab,length,doc,label = ds.load()
    num_class,vocab_size,emb_dim,hidden_dim,lr = args.num_clas,len(vocab),args.emb_dim,args.hidden_dim,args.lr
    model = TextRNN(num_class,vocab_size,emb_dim,hidden_dim,lr)
    train_x,test_x,train_y,test_y = train_test_split(doc,label,test_size=args.test_size)
    run(model,train_x,test_x,train_y,test_y,args.epoch)
if __name__ == '__main__':
    main()