import numpy as np

# Function used to load glove vectors
def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model


class W2V(object):

    def __init__(self, model, coco_caps):
        self.model = model
        self.coco_caps = coco_caps

    def get_text_representations(self, imgId, global_mean = False):

        annIds = self.coco_caps.getAnnIds(imgIds=imgId);
        anns = self.coco_caps.loadAnns(annIds)
        # Compute annotations mean (text word representation for one image)
        global_mean = False
        caption_vectors_mean = np.zeros(300)

        if global_mean:
            for i in range(len(anns)):
                caption_vectors = []
                for word in anns[i]['caption'][:-1].split(' '):
                    try:
                        caption_vectors.append(np.array(self.model[word.lower()]))
                        caption_vectors_mean += sum(caption_vectors)/len(caption_vectors)
                    except:
                        a = 1

            caption_vectors_mean = caption_vectors_mean/len(anns)

        # mean of one of the sentences
        else:
            caption_vectors = []
            for word in anns[0]['caption'][:-1].split(' '):
                try:
                    caption_vectors.append(np.array(self.model[word.lower()]))
                    caption_vectors_mean += sum(caption_vectors)/len(caption_vectors)
                except:
                    a = 1


        return caption_vectors_mean
