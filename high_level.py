import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap

silent = False          # Display track of progress info (when False)

class HighLevelClassification(object):

    def __init__(self, k, num_class, is_weighted = False, is_debug = False):
        self.k = k
        self.num_class = num_class
        self.is_weighted = is_weighted
        self.is_debug = is_debug
        self.is_fitted = False

        # Data structures required for each class
        self.nbrs = [NearestNeighbors(n_neighbors=self.k, metric='euclidean') for i in range(num_class)]   # Nearest Neighbors object
        # Radius Neighbors -> self.nbrs[class_id].radius
        self.G = [nx.Graph() for i in range(num_class)]
        self.original_index = [list() for i in range(num_class)]
        self.net_measure = [list() for i in range(num_class)]

        if self.is_debug: self.G_test = [nx.Graph() for i in range(num_class)] # for DEBUG only

    def fit(self, train_data, train_target):
        if self.is_fitted: # Clear previous trained model data
            self.__init__(self.k, self.num_class, is_weighted = self.is_weighted, is_debug = self.is_debug)
        # Build network for each class
        for class_id in range(self.num_class):
            train_target_class_idx = np.where(train_target==class_id)[0]
            train_data_class = train_data[train_target_class_idx]
            self.build_init_network(class_id, train_data_class, train_target_class_idx)

    def get_radius(self, distances):
        return np.median(distances[:,self.k-1])

    def calculate_measure(self, G):
        measures = []

        # Communicability Measure
        comm = nx.communicability_exp(G) # returns dictionary
        comm_list = [np.mean(list(i.values())) for i in list(comm.values())]
        avg_comm = np.mean(comm_list)
        measures.append(avg_comm)

        return measures

    def build_init_network(self, class_id, train_data_class, train_data_original_index):
        self.is_fitted = True
        # Setting/pointing proper data structures
        G_class = self.G[class_id]
        self.original_index[class_id] = train_data_original_index # This index match that passed for 
                                                                  # training and the ones used on networks 
                                                                  # but not the kNNs ones specific for
                                                                  # each class
        nbrs_class = self.nbrs[class_id]
        net_measure_class = self.net_measure[class_id]
        
        # Calculating Nearest Neighbors
        nbrs_class.fit(train_data_class)
        knn_distances, knn_indexes = nbrs_class.kneighbors() # Call for model, not instance
        # Calculating Radius Neighbors
        nbrs_class.set_params(radius = self.get_radius(knn_distances))
        radius_distances, radius_indexes = nbrs_class.radius_neighbors() # Call for model, not instance

        # Generating Nodes
        for index, instance in enumerate(train_data_class):
            G_class.add_node(int(train_data_original_index[index]), value=instance, typeNode="init_net")
        
        # Generating Edges
        for idx in range(len(train_data_class)): 
            current_node = int(train_data_original_index[idx]) # original index / index used in graph (but not in nbrs)
            if not silent: print('Connecting node '+str(idx+1)+' of ' + str(len(train_data_class)) + ' for class ' + str(class_id) + '.\r', end="")
            # Determine which (kNN or RN) method of connection
            if (len(radius_indexes[idx])) > self.k:  
            # RN connection : dense area
                nb_list = [int(train_data_original_index[nb]) for nb in radius_indexes[idx]]
                if self.is_weighted:
                    edge_list = [(current_node, nb, float(radius_distances[idx][i])) for i,nb in enumerate(nb_list)]
                    G_class.add_weighted_edges_from(edge_list)
                else:
                    edge_list = [(current_node, nb) for i,nb in enumerate(nb_list)]
                    G_class.add_edges_from(edge_list)
            else:
            # kNN connection : sparse area
                nb_list = [int(train_data_original_index[nb]) for nb in knn_indexes[idx]]
                if self.is_weighted:
                    edge_list = [(current_node, nb, float(knn_distances[idx][i])) for i,nb in enumerate(nb_list)]
                    G_class.add_weighted_edges_from(edge_list)
                else:
                    edge_list = [(current_node, nb) for i,nb in enumerate(nb_list)]
                    G_class.add_edges_from(edge_list)
        if not silent: print()

        self.merge_components(class_id)

        # Calculate Measures from "Original" Network (Trained)
        net_measure_class.append(self.calculate_measure(G_class))
        #print("Measures for class " + str(class_id) + ":", net_measure_class) # for DEBUG


    def merge_components(self, class_id):
        """
        Merge isolated components.
        The classification stage does not require merging, so this function is 
        required only to finish the training stage when the network is fragmented.
        """
        G_class = self.G[class_id]

        # Check Isolated Components
        Gcc = sorted(nx.connected_components(G_class))
        if len(Gcc) == 1:
            return
        
        if not silent: print('Class '+str(class_id)+' required MERGE!')
        while len(Gcc)>1:
            fixedComponent = Gcc[0]
            candidates = set.union(*Gcc[1:len(Gcc)])
            pairs = [(x,y) for x in fixedComponent for y in candidates]
            dist_list = [None] * len(pairs)
            
            for i, p in enumerate(pairs):
                d = np.linalg.norm(G_class._node[p[0]]['value'] - G_class._node[p[1]]['value'])
                dist_list[i] = d

            # Connecting closest pair
            idx = dist_list.index(min(dist_list))
            nodeA = pairs[idx][0]
            nodeB = pairs[idx][1]
            if self.is_weighted:
                G_class.add_edge(nodeA, nodeB, weight=float(dist_list[idx]))
            else:
                G_class.add_edge(nodeA, nodeB)

            Gcc = sorted(nx.connected_components(G_class), key=len, reverse=True) # recalculate components


    def predict(self, test_data):
        predicted_labels = [[] for i in range(len(test_data))]

        for idx, instance in enumerate(test_data):
            if not silent: print('Predicting sample '+str(idx+1)+' of ' + str(len(test_data)) + '.\r', end="")
            dist_list = []

            # Each class is analyzed independently
            # Insert node into each of the classes
            for class_id in range(self.num_class):
                G_class = self.G[class_id]
                nbrs_class = self.nbrs[class_id]
                original_index = self.original_index[class_id]
                net_measure_class_before = self.net_measure[class_id]

                G_class.add_node('test', values=instance, typeNode='test')

                # Choose connection rule
                radius_distances, radius_indexes = nbrs_class.radius_neighbors([instance]) # Call for instance, not model
                if (len(radius_indexes[0])) > self.k:
                    #nodes = [original_index[i] for i in radius_indexes[0]]
                    if self.is_weighted:
                        edge_list = [('test', int(original_index[nb]), float(radius_distances[0][i])) for i,nb in enumerate(radius_indexes[0])]
                        G_class.add_weighted_edges_from(edge_list)
                    else:
                        edge_list = [('test', int(original_index[nb])) for i,nb in enumerate(radius_indexes[0])]
                        G_class.add_edges_from(edge_list)
                else:
                    knn_distances, knn_indexes = nbrs_class.kneighbors([instance]) # Call for instance, not model
                    if self.is_weighted:
                        edge_list = [('test', int(original_index[nb]), float(knn_distances[0][i])) for i,nb in enumerate(knn_indexes[0])]
                        G_class.add_weighted_edges_from(edge_list)
                    else:
                        edge_list = [('test', int(original_index[nb])) for i,nb in enumerate(knn_indexes[0])]
                        G_class.add_edges_from(edge_list)

                # Calculate new network measures
                
                net_measure_class_after = self.calculate_measure(G_class)  
                V1, V2 = np.array(net_measure_class_before), np.array(net_measure_class_after)
                euclidean_dist = np.linalg.norm(V2 - V1)
                # Normalization (proportional disturbance)
                measure = euclidean_dist/float(V1)
                dist_list.append(measure)
                
                if self.is_debug: self.G_test[class_id] = copy.deepcopy(G_class) # DEBUG only
                G_class.remove_node('test') # delete node after measurement

            # Predict class by minimum disturbance between classes
            min_value = min(dist_list)
            predicted = int(dist_list.index(min_value))
            predicted_labels[idx] = predicted

        if not silent: print()
        
        return predicted_labels


    def score(self, test_data, test_target):
        predicted_labels = self.predict(test_data)
        correct_labels = list(map(int, test_target))
        hits = [predicted_labels[i]==correct_labels[i] for i in range(len(correct_labels))]
        score = round(sum(hits)/len(hits), 3)
        return score

    def accuracy_score(self, predicted_labels, test_target):
        correct_labels = list(map(int, test_target))
        print("original_label:", correct_labels)
        print("predict_label:", predicted_labels)

        hits = [predicted_labels[i]==correct_labels[i] for i in range(len(correct_labels))]
        accuracy = round(sum(hits)/len(hits), 3)

        print("Correct number:", sum(hits))
        print("Accuracy:", accuracy)
            
        return accuracy

    # Plot Network Formed with High-Level Technique
    def plotNet(self, X_train, y_train, X_test = None, ax = None, cmap = None, mode = 'train'):
        assert mode == 'train' or mode == 'test'
        original_positions = {item:X_train[item] for item in range(len(X_train))}
        if mode == 'train':
            F = nx.compose(*self.G)
            y_c = y_train
        elif mode == 'test':
            assert len(X_test)==1 # Test network is evaluated 1-by-1 (sample)
            original_positions['test'] = np.array(*X_test)
            predicted_label = int(self.predict(X_test)[0])
            net_list = copy.copy(self.G)
            net_list[predicted_label] = self.G_test[predicted_label]
            F = nx.compose(*net_list)
            re_idx = list(F.nodes)
            re_idx[re_idx.index('test')] = len(re_idx)-1
            y_c_temp = np.append(y_train, [0.5]) # Test point is hard-marked with 0.5
            y_c = [y_c_temp[i] for i in re_idx]
        if ax == None:
            ax = plt.figure(figsize=(10, 10))
            cmap = plt.cm.Set1
            nx.draw_networkx(F, original_positions, with_labels=False, node_size=plt.rcParams['lines.markersize']**2, node_color=y_c, cmap=cmap, width=0.2, alpha=0.6)
            plt.pause(0.01)
        else:
            nx.draw_networkx(F, original_positions, with_labels=False, ax=ax, node_size=plt.rcParams['lines.markersize']**2, node_color=y_c, cmap=cmap, width=0.2, alpha=0.6)