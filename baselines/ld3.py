import numpy as np
import pandas as pd 

from skmultilearn.dataset import load_dataset

from skmultiflow.data import MultilabelGenerator, ConceptDriftStream

from sklearn.metrics import f1_score
from sklearn import metrics

from river.base import DriftDetector



def find_anomalies(data, coeff=3):
    #define a list to accumlate anomalies
    anomalies = []
    
    # Set upper and lower limit to 3 standard deviation
    random_data_std = np.std(data)
    random_data_mean = np.mean(data)
    anomaly_cut_off = random_data_std * coeff
    
    lower_limit  = random_data_mean - anomaly_cut_off 
    upper_limit = random_data_mean + anomaly_cut_off
    #print(lower_limit)
    # Generate outliers
    for outlier in data:
        if outlier < lower_limit: # or outlier > upper_limit:
            anomalies.append(outlier)
    return anomalies

def induce_drift(X, Y, start, end, num_labels, num_features, percentage=10):
    for i in range(len(Y)):
        #indexes = []
        if i >= start and i < end:
            #index = np.random.randint(0, num_labels)
            two_change = np.random.randint(0, 2) == 1
            '''while Y[i][index] != 0:
                index = np.random.randint(0, num_labels)
            #indexes.append(index)'''
            Y[i][0] = 1

            if two_change:
                '''index = np.random.randint(0, num_labels)
                while Y[i][index] != 0:
                    index = np.random.randint(0, num_labels)
                #indexes.append(index)'''
                Y[i][1] = 1

    num_changes = num_features // percentage
    for i in range(len(X)):
        indexes = []
        if i >= start and i < end:
            for j in range(num_changes):
                '''index = np.random.randint(0, num_features)
                while X[i][index] != 0 and index not in indexes:
                    index = np.random.randint(0, num_labels)
                indexes.append(index)'''
                if X[i][j] == 0:
                    X[i][j] = 1
                else:
                    X[i][j] = 0

        
    return X, Y

def WS(r1,r2):
        sum_ = 0
        ranks_x = np.argsort(r1)
        ranks_y = np.argsort(r2)
        for i in range(len(r1)):
            sum_ += (1/(2**(ranks_x[i]))) * ((np.abs(ranks_x[i] - ranks_y[i]))/(np.max([np.abs(1-ranks_x[i]), np.abs(len(r1) - ranks_x[i])])))
        return 1 - sum_

def generate_csv(dataset, save_dir='./', split='train', self_loops=False):
    X_train, y_train, _, _ = load_dataset(dataset, split)
    y_train = y_train.toarray()
    X_train = X_train.toarray()


    features = ["{}".format(i) for i in range(X_train.shape[1])]
    labels = ["{}".format(i+X_train.shape[1]) for i in range(y_train.shape[1])]
    
    X = pd.DataFrame(data=X_train, columns=features)
    X.index.name = 'index'
    Y = pd.DataFrame(data=y_train, columns=labels)
    X.index.name = 'index'
    
    df = pd.merge(X, Y, right_index=True, left_index=True)
    df.to_csv('./{}/{}.csv'.format(save_dir, dataset), sep='\t', encoding='utf-8')

def load_data(filename, in_dir='./', dtype='dataframe'):
    '''rows = None
    with open('{}/scene_ranking.txt'.format(in_dir), 'r') as f:
        rows = f.readlines()
    A = [literal_eval(row.strip()) for row in rows]
    A = [[r[1] for r in row] for row in A]
    A = np.matrix(A)
    return A'''
    df = pd.read_csv('{}/{}'.format(in_dir, filename), sep='\t', index_col='index')
    
    if dtype == 'dataframe':
        return df
    else:
        return np.matrix(df.values)
def to_numpy_matrix(Y, self_loops=False):
    '''num_labels = Y.shape[1]
    freqs = np.matrix(np.zeros(shape=(num_labels, num_labels), dtype=np.int32))
    for label in Y:
        l = np.matrix(label)
        a = np.matmul(np.transpose(l), l, dtype=np.int32)
        freqs = np.add(freqs, a)'''
    t = np.transpose(Y)
    freqs = t@Y
    if not self_loops:    
        np.fill_diagonal(freqs, 0)
    #freqs = np.matrix(freqs)
    return freqs

def w(r):
    if r < 2:
        return r
    else:
        return 0

def accuracy_example(y_true, y_pred):
    N = len(y_true)
    sum_ = 0
    for i in range(N):
        nom = np.logical_and(y_true[i], y_pred[i]).sum()
        denom = np.logical_or(y_true[i], y_pred[i]).sum()
        sum_ += nom/denom

    return sum_ / N

def hamming_loss(y_true, y_pred):
    return metrics.hamming_loss(y_true, y_pred)

def f1_example(y_true, y_pred):
    p = precision_example(y_true, y_pred)
    r = recall_example(y_true, y_pred)

    return (p * r) / ((2 * p) + r)

def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def precision_example(y_true, y_pred):
    N = len(y_true)
    sum_ = 0
    for i in range(N):
        nom = np.logical_and(y_true[i], y_pred[i]).sum()
        denom = len(y_pred[i])
        sum_ += nom/denom
    
    return sum_ / N

def recall_example(y_true, y_pred):
    N = len(y_true)
    sum_ = 0
    for i in range(N):
        nom = np.logical_and(y_true[i], y_pred[i]).sum()
        denom = len(y_true[i])
        sum_ += nom/denom
    
    return sum_ / N

def recip_rank(mat, self_loops=False):
    num_labels = len(mat)
    ranks = np.flip(np.argsort(mat), axis=1)
    if not self_loops:
        ranks = np.delete(ranks, [np.argwhere(i == ranks[i]).flatten()[0]+i*num_labels for i in range(num_labels)]).reshape(num_labels,num_labels-1)
    return np.argsort([1/np.sum(1/(np.argwhere(ranks==i)[:, 1] + 1)) for i in range(num_labels)])

def add_to_pretrain(X, y, sample, label, size):
    if len(X) < size:
        X.append(sample)
        y.append(label)
    else:
        X.pop(0)
        X.append(sample)
        y.pop(0)
        y.append(label)
    return X, y 


def add_left_shift_drift(y, start_point, shift_amount=2):
    new_y = []
    for i, label in enumerate(y):
        label = label.tolist()
        if i >= start_point:
            for i in range(shift_amount):
                elem = label.pop(0)
                label.append(elem)
        new_y.append(np.array(label))
    return np.array(new_y)

def add_label_count_drift(y, start_point, count=3):
    new_y = []
    for i, label in enumerate(y):
        label = label.tolist()
        if i >= start_point and i % 3 == 0:
            for j in range(count):
                label[j] = 1
        new_y.append(label)
    new_y.reverse()
    return np.array(new_y)


class LD3(DriftDetector):
    def __init__(self, window_size = 500, correlation_thresh=3, len=2):
        super().__init__()
        self.window_size = window_size
        self.len = len
        self.w1 = Window(max_size=window_size)
        self.w2 = Window(max_size=window_size)
        self.w3 = Window(max_size=window_size)
        self.warmup = True
        self.warmup_count = window_size * 2
        self.r1 = None
        self.r2 = None
        self.correlation_thresh = correlation_thresh
        self.past_correlation = None
        self.prev_correlation = None

    def update(self, value):
        self.insert_element(value.flatten())
        if not self.warmup:
            self.r1 = recip_rank(to_numpy_matrix(self.w1.get_window, self_loops=False), self_loops=False)
            self.r2 = recip_rank(to_numpy_matrix(self.w2.get_window, self_loops=False), self_loops=False)


            drift, warning = False, False
            correlation= self.WS()
            

            if self.w3.len < self.window_size:
                self.w3.queue(correlation)
                return False, False, 0, 0

            score = find_anomalies(self.w3.get_window, coeff=self.correlation_thresh)
            self.w3.queue(correlation)
            if len(score) > self.len and correlation < 1: 
                drift = True
                warning = True
                

            if drift:
                values = np.asarray([value for value in np.linspace(-1,1,100)])
                values = values.reshape((len(values), 1))
                self.clear_windows()
                self.warmup_counter(warmup_count=self.window_size*2)

 
            
            self.prev_correlation = correlation

            return drift, warning, correlation, score

        return False, False, 0, 0


    def WS(self):
        sum_ = 0
        ranks_x = np.argsort(self.r1)
        ranks_y = np.argsort(self.r2)
        for i in range(len(self.r1)):
            sum_ += (1/(2**(ranks_x[i]))) * ((np.abs(ranks_x[i] - ranks_y[i]))/(np.max([np.abs(1-ranks_x[i]), np.abs(len(self.r1) - ranks_x[i])])))
        return 1 - sum_
        

    def warmup_counter(self, warmup_count=None):
        if self.warmup_count > 0:
            self.warmup_count -= 1
        else:
            self.warmup = False

        if warmup_count is not None:
            self.warmup_count = warmup_count
            self.warmup = True

    def insert_element(self, value):
        self.warmup_counter()
        v = self.w1.queue(value)
        if v is not None:
            u = self.w2.queue(v)
    
    def clear_windows(self):
        self.w1.clear()
        self.w2.clear()
        self.w3.clear()

    
    @property
    def _ranks(self):
        return self.r1, self.r2
    


class Window():
    def __init__(self, max_size=250):
        self.max_size = max_size
        self.size = 0
        self.window = []
    
    def queue(self, value):
        if self.size <= self.max_size:
            self.window.append(value)
            self.size += 1
            popped = None
        else:
            popped = self.dequeue()
            self.window.append(value)
        return popped
    
    def dequeue(self):
        return self.window.pop(0)

    def clear(self):
        self.size = 0
        self.window.clear()

    def increase_size(self, value):
        self.max_size = value
    
    def decrease_size(self, value):
        self.max_size = value
    
    def set_window(self, window):
        self.window = window
    
    @property
    def get_window(self):
        return np.array(self.window)
    
    @property 
    def len(self):
        return len(self.window)

class StreamGenerator():
    def __init__(self):
        pass

    def get_stream(self, type):
        sample_size = 0
        stream = None
        n_features = 0
        n_targets = 0
        if type == 'sudden1':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=25, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=25, random_state=0)
            sample_size=20000
        elif type == 'gradual1':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=250, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=250, random_state=0)
            sample_size=20000
        elif type == 'sudden2':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'sudden3':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=50, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=50, random_state=0)
            sample_size=20000
        elif type == 'mixed':
            n_features = 100
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=25000, n_features=n_features, n_targets=n_targets, n_labels=1, random_state=100) 
            s2 = MultilabelGenerator(n_samples=25009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=250)
            s3 = MultilabelGenerator(n_samples=25015, n_features=n_features, n_targets=n_targets, n_labels=3, random_state=0)
            s4 = MultilabelGenerator(n_samples=25015, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100)
            s5 = MultilabelGenerator(n_samples=25009, n_features=n_features, n_targets=n_targets, n_labels=1, random_state=250)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=20000, width=1, random_state=0)
            stream2 = ConceptDriftStream(stream=stream1, drift_stream=s3, position=40000, width=100, random_state=0)
            stream3 = ConceptDriftStream(stream=stream2, drift_stream=s4, position=60000, width=100, random_state=0)
            stream = ConceptDriftStream(stream=stream3, drift_stream=s5, position=80000, width=1, random_state=0)
            sample_size=100000
        elif type == 'gradual2':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=500, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=500, random_state=0)
            sample_size=20000
        elif type == 'gradual3':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1000, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1000, random_state=0)
            sample_size=20000
        elif type == 'reoccurring':
            n_features = 200
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0) 
            s2 = MultilabelGenerator(n_samples=6500, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=15500, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=500, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=500, random_state=0)
            sample_size=20000
        elif type == 'benchmark1':
            n_features = 100
            n_targets = 20
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'benchmark2':
            n_features = 100
            n_targets = 50
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        elif type == 'benchmark3':
            n_features = 100
            n_targets = 100
            s1 = MultilabelGenerator(n_samples=6000, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=100) 
            s2 = MultilabelGenerator(n_samples=6009, n_features=n_features, n_targets=n_targets, n_labels=4, random_state=250)
            s3 = MultilabelGenerator(n_samples=10009, n_features=n_features, n_targets=n_targets, n_labels=2, random_state=0)
            stream1 = ConceptDriftStream(stream=s1, drift_stream=s2, position=4000, width=1, random_state=0)
            stream = ConceptDriftStream(stream=stream1, drift_stream=s3, position=10000, width=1, random_state=0)
            sample_size=20000
        return stream, sample_size, n_features, n_targets