import fnmatch
import os
import random
import re
import threading
import time
import datetime
import csv
import numpy as np
import tensorflow as tf
import math

#NONSTATIONARY = [0, 1, 4, 5, 9, 10]
FEATURES_PRICES = 11
#FEATURES_PRICES = 4
#FEATURES_ORDERS = 29
FEATURES_ORDERS = 10
FEE = 0.0

#PREDICTION_LAG = 360    
    
def find_dirs(directory):
    dirs = []
    for root, dirnames, filenames in os.walk(directory):
        for subdir in dirnames:
            t = time.mktime(datetime.datetime.strptime(subdir, "%d-%m-%Y_%H:%M:%S").timetuple())
            dirs.append(t)
    dirs.sort()
    finaldirs = []
    for d in dirs:
        name = datetime.datetime.fromtimestamp(int(d)).strftime('%d-%m-%Y_%H:%M:%S')
        finaldirs.append(os.path.join(directory, name))
    return finaldirs

     
def fill_missing_avgprices(data):
    for row in data:
        if FEATURES_PRICES < 6:
            return
        if row[5] == 0:
            row[5] = row[2]
        if row[6] == 0:
            row[6] = row[1]
            
            
def apply_differencing(data):
    updated = []
    for index in range(1, len(data) - 1):
        row = data[index]
        previousRow = data[index - 1]
        updatedRow = []
        for i in xrange(len(row)):
            updatedRow.append(row[i])  
        if FEATURES_PRICES > 6:
            # write avg prices and prices from other exchanges as diffs
            updatedRow[4] -= row[1]
            updatedRow[5] -= row[0]
            midPrice = (float(row[0]) + float(row[1])) / 2
            updatedRow[9] -= midPrice
            updatedRow[10] -= midPrice
        # first differencing prices
        updatedRow[1] -= updatedRow[0]
        updatedRow[0] -= previousRow[0]       
        updated.append(updatedRow)
    return updated


def single_csv_import(path, result, numColumns):
    overwrite = len(result) > 0
    firstDate = 0
    lastDate = 0
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        rowIndex = 0
        for row in reader:
            u = []
            colIndex = 0
            while colIndex < numColumns:
                if len(row) > colIndex:
                    u.append(float(row[colIndex]))
                else:
                    return False        
                colIndex += 1
            if overwrite:
                if rowIndex < len(result):
                    result[rowIndex].extend(u)
            else:
                result.append(u)
            rowIndex += 1
    return True


def start_thread(func, sess, test_every):
    thread = threading.Thread(target=func, args=(sess,test_every))
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()
    return thread


def load_from_files(dirs, index):
    result = []   
    res1 = single_csv_import(dirs[index] + "/prices.csv", result, FEATURES_PRICES + 1)       
    res2 = single_csv_import(dirs[index] + "/bids.csv", result, FEATURES_ORDERS)
    res3 = single_csv_import(dirs[index] + "/asks.csv", result, FEATURES_ORDERS)
    if res1 and res2 and res3:
        return result
    else:
        return None


def load_and_augment_from_files(dirs, index, receptive_field, prediction_lag):
    result = load_from_files(dirs, index)
    firstDate = result[0][0]
    lastDate = result[-1][0]
    if result is not None:
        if index > 0:
            prevResult = load_from_files(dirs, index - 1)
            prevLastDate = prevResult[-1][0]
            if prevResult is not None and abs(prevLastDate - firstDate) < 300:
                result = prevResult[-receptive_field:] + result
        
        if index < len(dirs) - 1:
            nextResult = load_from_files(dirs, index + 1)
            nextFirstDate = nextResult[0][0]
            if nextResult is not None and abs(nextFirstDate - lastDate) < 300:
                result = result + nextResult[:prediction_lag]
    return result


class DataReader(object):
    '''Generic background data reader that preprocesses price files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 testdata_dir,
                 coord,
                 receptive_field,
                 prediction_lag = 360,
                 quantization_thresholds = [0.01, 0.025],
                 sample_size = None,
                 queue_size = 10):
        self.data_dir = data_dir
        self.testdata_dir = testdata_dir
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.prediction_lag = prediction_lag
        self.quantization_thresholds = quantization_thresholds
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, self.num_features() + 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.cache = [None] * 10
        self.test_cache = [None] * 10
        self.category_cardinalities = [0] * (self.num_return_categories())

        dirs = find_dirs(data_dir)
        if not dirs:
            raise ValueError("No directories found in '{}'.".format(data_dir))
        self.compute_statistics(dirs)
        
        
            
    def num_features(self):
        return FEATURES_PRICES + 2 * FEATURES_ORDERS
    
    def num_return_categories(self):
        return len(self.quantization_thresholds) * 2 + 1
    
    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output
    
    
    def thread_main(self, sess, test_every):
        stop = False
        # Go through the dataset multiple times
        counter = 0
        while not stop:
            iterator = self.load_data()        
            for data in iterator:
                if self.coord.should_stop():
                    stop = True
                    break           
                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    pieces = []
                    while len(data) > self.receptive_field + 2 * self.sample_size:
                        piece = data[:(self.receptive_field +
                                        self.sample_size), :]
                        pieces.append(piece)                
                        data = data[self.sample_size:, :]
                    
                    pieces.append(data)
                    for piece in pieces:
                        index = random.randint(0, (len(pieces) - 1))
                        #print(index)
                        sess.run(self.enqueue,
                            feed_dict={self.sample_placeholder: pieces[index]})                       
                        if counter % test_every == 0:                          
                            self.enqueue_test(sess)
                        counter += 1
                            
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: data})
                             
                    #print(data[0:5, :])
                    
                    if counter % test_every == 0:
                        self.enqueue_test(sess)
                    counter += 1
    
    
    def enqueue_test(self, sess):
        iterator_test = self.load_data(True)
        test_data = next(iterator_test)
        sess.run(self.enqueue,
            feed_dict={self.sample_placeholder: test_data})


    def start_threads(self, sess, test_every):
        self.threads.append(start_thread(self.thread_main, sess, test_every))
        return self.threads
        
        
    def load_data(self, test = False):
        dirs = find_dirs(self.data_dir)
        if test:
            testDirs = find_dirs(self.testdata_dir)
            #use training data only to fill the receptive field
            dirs.append(testDirs[0])
            data = self.load_from_csv_or_cache(dirs, len(dirs) - 1, test)
            #print(data[:5,:])
            #print(data[-5:,:])
            yield data
        else:           
            for dir in dirs:
                index = random.randint(0, (len(dirs) - 1))
                data = self.load_from_csv_or_cache(dirs, index, test)
                if data is not None:
                    yield data
       
        data_dir = self.testdata_dir if test else self.data_dir
        
                
    
    def compute_statistics(self, dirs):
        self.runningStats = [0, [[0.0, 0.0] for _ in xrange(self.num_features())]]       
        for i in range(0, len(dirs)):
            data = self.load_from_csv(dirs, i, True)
            if data is None:
                print("Batch discarded")
        
               
    def load_from_csv(self, dirs, index, onlyUpdateStats = False):
        result = load_and_augment_from_files(dirs, index, self.receptive_field, self.prediction_lag)
        if result is not None:
            fill_missing_avgprices(result)         
            result = self.add_target(result, onlyUpdateStats)
            #print(result[1])
            #result = result[:][1:]  #take out time column
            result = [row[1:] for row in result]
            #print(result[1])
            result = apply_differencing(result)
            if onlyUpdateStats:
                self.update_statistics(result)
            else:
                self.normalize_features(result)
            result = np.array(result)
            #print("First date: {}, last date: {}".format(res1[1], res1[2]))
        else:
            result = None
        return result
    
    
    def load_from_csv_or_cache(self, dirs, index, test):
        cache = self.test_cache if test else self.cache
        if index < len(cache) and cache[index] is not None:
            result = cache[index]
        else:
            result = self.load_from_csv(dirs, index)
            if index < len(cache):
                cache[index] = result
        return result
        
        
    def update_statistics(self, result):
        currentDim = self.runningStats[0]
        meansAndVars = self.runningStats[1]
        for row in result:
            #print(row[1])
            for i in range(0, self.num_features()):
                mean = meansAndVars[i][0]
                meansAndVars[i][0] += (row[i] - mean) / (currentDim + 1)
                meansAndVars[i][1] += (row[i] - mean) * (row[i] - meansAndVars[i][0])
            currentDim += 1
        self.runningStats[0] = currentDim
      
    
    def normalize_features(self, result):
        dim = self.runningStats[0]
        meansAndVars = self.runningStats[1] 
        #print(meansAndVars)
        for i in range(0, self.num_features()):
            mean = meansAndVars[i][0]
            stdev = math.sqrt(meansAndVars[i][1] / (dim - 1))       
            for row in result:
                row[i] = (row[i] - mean) / stdev
                
        
    def add_target(self, data, onlyUpdateStats):
        buckets = [0] * (self.num_return_categories())
        lag = self.prediction_lag

        for index in range(0, len(data) - lag):
            actualSell = data[index][1]
            actualBuy = data[index][2]
            
            effective_lag = lag
            
            while data[index + effective_lag][0] - data[index][0] > lag * 60:
                effective_lag -= 1
            while index + effective_lag < len(data) and data[index + effective_lag][0] - data[index][0] < lag * 60:
                effective_lag += 1
            if index + effective_lag >= len(data):
                break
            
            futureSell = data[index + effective_lag][1]
            futureBuy = data[index + effective_lag][2]

            longReturn = futureSell / actualBuy - FEE - 1
            shortReturn = - futureBuy / actualSell - FEE + 1
            if longReturn > shortReturn:
                category = self.quantize_return(longReturn, True)
            else:
                category = self.quantize_return(shortReturn, False)
            data[index].append(category)
            buckets[category] += 1
        if onlyUpdateStats:
            print("Return buckets: {}".format(buckets))
            self.category_cardinalities = [sum(x) for x in zip(self.category_cardinalities, buckets)]
        return data[:-lag]
     
    
    def get_category_inv_weights(self):
        total = sum(self.category_cardinalities)
        result = [round(1 - float(x)/total, 2) for x in self.category_cardinalities]
        print("Inverse category weights: {}".format(result))
        return result
        
        
    def quantize_return(self, ret, buy):
        mult = 1
        if not buy:
            mult = -1
        cat = 0
        n = len(self.quantization_thresholds)
        while cat < n and ret > self.quantization_thresholds[cat]:
            cat += 1
        cat = cat * mult + n
        return cat
        

