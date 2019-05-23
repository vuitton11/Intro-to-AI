# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
  """
  Perceptron classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
    self.extra = True

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details.

    Use the provided self.weights[label] data structure so that
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """

    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
    acc = 0
    for iteration in range(self.max_iterations):
        print "Executing iteration ", iteration + 1, "..."
        for i in range(len(trainingData)):
            "*** YOUR CODE HERE ***"
            target = None
            label = None

            for x in self.legalLabels:
                count = 0
                for f, num in trainingData[i].items():
                    count += num * self.weights[x][f]
                #print count

                #output the scoring
                if target is None:
                    target = count
                    #print target

                    label = x
                    #print label

                elif count > target:
                    target = count
                    # print target

                    label = x
                    # print label
            t = trainingLabels[i]

            if label == t:
                tempp = "hit"
                #print 'match'
            else:
                self.weights[t] = self.weights[t] + trainingData[i]
                #print "working"
                self.weights[label] = self.weights[label] - trainingData[i]
                #print "working"


        val = self.classify(validationData)
        #print val

        len1 = len(validationLabels)
        ran = range(len1)
        superCount = 0
        for j in ran:
            if val[j] == validationLabels[j]:
                superCount = superCount +1
        #print superCount
        print "i:", iteration, " accuracy:", (100.0 * superCount / len1)

        #switch accordingly
        if superCount > acc:
            newW = self.weights.copy()
            acc = superCount

    self.weights = newW


  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.

    Recall that a datum is a util.counter...
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses


  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    temp = self.weights[label]
    #print temp

    featuresWeights = temp.sortedKeys()[:100]
    #print featuresWeights

    return featuresWeights
