from scipy.linalg import eigh, norm
import numpy as np
import itertools
from scipy import spatial
from scipy.misc import imread, imresize
import tensorflow as tf
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# This class performs all the search tasks
class Search(object):

    def __init__(self, model, coco_caps, vgg, sess):
        self.data_c = []
        self.W = []
        self.e_values = []
        self.model = model
        self.coco_caps = coco_caps
        self.vgg = vgg
        self.sess = sess

    def T2I(self, sentence, neighbors_number = 5, method = 'similarity'):
        #method is either 'similarity' or 'distance'
        #generate the sentence vector
        all_words = sentence.split()
        list_vectors = np.zeros((len(all_words), 300))
        count = 0
        for word in sentence.split():
            list_vectors[count, :] = self.model[word]
            count+=1

        sentence_vector = np.mean(list_vectors, axis = 0)
        tag_vector = np.dot(sentence_vector.T, self.W[0])

        return self.findClosestNeighbors(tag_vector, neighbors_number, 'T2I', method)

    def I2T(self, id_query, features, IMAGES_PATH, neighbors_number = 5, cat = False ,method = 'similarity'):

        dic_features = {'fc1': self.vgg.fc1, 'fc2': self.vgg.fc2, 'fc3l': self.vgg.fc3l}
        imgIds = self.coco_caps.getImgIds()
        #method is either 'similarity' or 'distance'
        #generate the sentence vector
        img = self.coco_caps.loadImgs(imgIds[id_query])[0]

        #image Features
        img1 = imread(IMAGES_PATH+img['file_name'], mode='RGB')
        img1 = imresize(img1, (224, 224))
        prob = self.sess.run(dic_features[features], feed_dict={self.vgg.imgs: [img1]})[0]
        image_vector = np.dot(prob.T, self.W[1])

        if cat:
            return self.findClosestNeighbors(image_vector, neighbors_number, 'I2C', method)
        else:
            return self.findClosestNeighbors(image_vector, neighbors_number, 'I2T', method)

    # returns the images and captions from the result closest_ids and plots and saves the plot
    def getT2Iresults(self, closest_ids, filename, save = True, display = True):
        imgIds = self.coco_caps.getImgIds()
        square_side = int(np.sqrt(len(closest_ids)))
        fig = plt.figure(figsize=(18, 18))
        gs = gridspec.GridSpec(square_side, square_side, wspace=0.0, hspace=0.0)
        result_images = []
        result_captions = []
        for i in range(len(closest_ids)):
            img = self.coco_caps.loadImgs(imgIds[closest_ids[i]])[0]

            #Get the image
            I = io.imread('images/train2014/%s'%(img['file_name']))
            I = imresize(I, (224, 224))
            result_images.append(I)

            #Get the captions
            annIds = self.coco_caps.getAnnIds(imgIds=img['id']);
            anns = self.coco_caps.loadAnns(annIds)
            result_captions.append(anns)

            ax = plt.subplot(gs[int(i//square_side), int(i%square_side)])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(I)
            fig.add_subplot(ax)
        if save:
            plt.savefig(filename, bbox_inches='tight')
        if display:
            plt.show()

        return result_images, result_captions


    def findClosestNeighbors(self, vector, neighbors_number, type_of_search, method):

        if type_of_search == 'T2I':
            n = 1
        elif type_of_search == 'I2T':
            n = 0
        elif type_of_search == 'I2C':
            n = 2
        else:
            print("unvalid type search: either 'T2I' or 'I2T'")

        #get nearest neighbors with euclidean distance:
        if method == 'distance':
            tree = spatial.KDTree(self.data_c[n])
            closest_ids = tree.query(vector, k=neighbors_number)[1]

        #get nearest neighbors with similarity
        elif method == 'similarity':
            D = np.diag([element**4 for element in self.e_values])
            similarities = np.zeros((self.nSamples, 1))
            for k in range(self.nSamples):
                M_i = vector.dot(D)
                M_j = self.data_c[n][k].dot(D)
                similarities[k, 0] = M_i.dot(M_j.T)/(norm(M_i, ord=2)*norm(M_j, ord=2))

            ind = np.argpartition(similarities[:, 0], -neighbors_number)[-neighbors_number:]
            closest_ids = ind[np.argsort(similarities[:, 0][ind])][::-1]

        else:
            closest_ids = "invalid method"

        return closest_ids

#This class performs the algorithmic work
class CCA(Search):

    def __init__(self, model, coco_caps, vgg, sess, reg = 0, n_components = 50):
        super(CCA, self).__init__(model, coco_caps, vgg, sess)
        self.reg = reg
        self.n_components = n_components

    def train(self, data):
        self.nSamples = data[0].shape[0]

        dimensionsData = [d.shape[1] for d in data]

        # computes covariance matrices
        covariance_matrices = [np.dot(k.T, h) for k in data for h in data]


        # Compute left and right part of the generalized eigenvalues problem
        S_left_part = np.zeros((sum(dimensionsData), sum(dimensionsData)))
        S_right_part = np.zeros((sum(dimensionsData), sum(dimensionsData)))


        last_dimension_row = 0
        for i in range(len(dimensionsData)):
            last_dimension_column = 0

            S_right_part[last_dimension_row : last_dimension_row+dimensionsData[i],
                         last_dimension_row : last_dimension_row+dimensionsData[i]] = covariance_matrices[i*len(dimensionsData)+i]+self.reg*np.eye(dimensionsData[i])

            for j in range(len(dimensionsData)):

                S_left_part[last_dimension_row : last_dimension_row+dimensionsData[i],
                            last_dimension_column : dimensionsData[j]+last_dimension_column] = covariance_matrices[i*len(dimensionsData)+j]

                last_dimension_column += dimensionsData[j]

            last_dimension_row += dimensionsData[i]


        # Solve the eigenvalue problem (we solve it for the n_components highest eigenvalues)
        n_components_max = S_left_part.shape[0]
        self.e_values, e_vectors = eigh(S_left_part, S_right_part, eigvals = (n_components_max-self.n_components, n_components_max-1))

        # Compute representation in common space
        last_dimension = 0

        for i in range(len(dimensionsData)):
            self.W.append(e_vectors[last_dimension : last_dimension + dimensionsData[i]])
            self.data_c.append(np.dot(data[i],self.W[i]))

            last_dimension+=dimensionsData[i]

    def test(self, data):
        self.nSamples = data[0].shape[0]
        self.data_c = []
        dimensionsData = [d.shape[1] for d in data]

        last_dimension = 0
        for i in range(len(dimensionsData)):
            self.data_c.append(np.dot(data[i],self.W[i]))
            last_dimension+=dimensionsData[i]


    #Cross validate over reg and n_component value, train the model on the best set
    def CV(self, regs_CV, n_components_CV, data, method = 'scale'):

        self.scores = {}

        min_score = np.inf
        count = 0

        #main loop over all parameters
        for n in n_components_CV:
            for reg_CV in regs_CV:
                self.reg = reg_CV
                self.n_components = n
                self.train(data)
                #the value 4 for the power is the one suggested by the paper
                D = np.diag([element**4 for element in self.e_values])
                score = 0

                #compute score for this combination of parameters
                for k in range(self.nSamples):
                    for dimensions in itertools.combinations(range(len(self.data_c)), 2):
                        [i, j] = dimensions
                        #euclidian
                        if method == 'eucl':
                            score += distance.euclidean(self.data_c[i][k, :],self.data_c[j][k, :])
                        #scale
                        elif method == 'scale':
                            M_i = data[i][k].dot(self.W[i]).dot(D)
                            M_j = data[j][k].dot(self.W[j]).dot(D)
                            score += M_i.dot(M_j.T)/(norm(M_i, ord=2)*norm(M_j, ord=2))

                #save scores
                self.scores[(reg_CV, n)] = score

                #display computation progress
                count+=1
                if count%10 == 0:
                    print("Done " +str(count)+ " tests")

                #select best score
                if score < min_score:
                    reg_best = reg_CV
                    n_components_best = n
                    min_score = score

        #train with best parameters
        self.reg = reg_best
        self.n_components = n_components_best

        self.train(data)
