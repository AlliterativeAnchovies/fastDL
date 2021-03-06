import xml.etree.ElementTree as ET
import csv
from fastai.text import *
from fastai.dataset import *
from fastai.conv_learner import *
from fastai import lm_rnn
from fastai.structured import *
from fastai.column_data import *
import matplotlib.pyplot as plt
import PIL
import spacy
import html
import subprocess
import os
import shutil
import gzip

spacy.load("en")
re1 = re.compile(r'  +')

def loadTextFile(fileName):
  toReturn = Path(fileName).open('r').read()#!cat {fileName} >/dev/null
  return toReturn

def bashLS(directory):
  return subprocess.check_output(["ls","-1",directory],shell=False).decode("utf-8").split("\n")[:-1]

def bashCP(cpFrom,cpTo):
  if os.path.isdir(cpFrom):
    if os.path.exists(cpTo):
        shutil.rmtree(cpTo)
    shutil.copytree(cpFrom,cpTo)
  else:
    shutil.copy(cpFrom,cpTo)

def bash(inputs,shell=True):
  return subprocess.check_output(inputs,shell=shell)

def makeDirectory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def mimicCDOnPath(path,cdInput):
  return os.path.join(path,cdInput)

def createModelDirectory(name,trainingDataPath = "",validationDataPath = ""):
  #This function creates a new directory called "name", with subdirectories:
  # "trainingData", "validationData", "models", "csvs", "misc"
  #All are self explanatory (example: "csvs" holds saved csvs), except perhaps
  # "misc", which is just a catch-all for things that do not belong in other
  # subdirectories.
  #trainingDataPath and validationDataPath are expected to be either .tar,
  #.tar.gz, .zip, or a folder.
  #Folders should be passed with a /. at the end
  #This function should work in other cases as well, but it is not
  #guaranteed.  If they are empty strings, this function will ignore them.
  print("Making directories...")
  makeDirectory(name)
  extraPath = name
  makeDirectory(f"{extraPath}/trainingData")
  makeDirectory(f"{extraPath}/validationData")
  makeDirectory(f"{extraPath}/models")
  makeDirectory(f"{extraPath}/csvs")
  makeDirectory(f"{extraPath}/misc")
  if trainingDataPath != "":
    print(f"copying {trainingDataPath}")
    bashCP(f"{trainingDataPath}",f"{extraPath}/trainingData")
    extraPath = mimicCDOnPath(extraPath,"trainingData")
    if trainingDataPath[-3:] == ".gz":
      print(f"unzipping {trainingDataPath}")
      try:
        bash(["gzip",f"{extraPath}/{trainingDataPath}","-d",f"{extraPath}/{'/'.join((trainingDataPath.split('/')[:-1]))}"],shell=False)
      except:
        pass #for some stupid reason, gzip works but throws an error
      trainingDataPath = trainingDataPath[:-3]
    if trainingDataPath[-4:] == ".tar":
      print(f"unzipping {trainingDataPath}")
      bash(["tar","xopf",f"{extraPath}/{trainingDataPath}","--directory",f"{extraPath}/{'/'.join((trainingDataPath.split('/')[:-1]))}"],shell=False)
    if trainingDataPath[-4:] == ".zip":
      print(f"unzipping {trainingDataPath}")
      bash(["unzip","-o","-d",f"{extraPath}/{'/'.join((trainingDataPath.split('/')[:-1]))}","-qq",f"{extraPath}/{trainingDataPath}"],shell=False)
    if trainingDataPath[-4:] == ".tgz":
      print(f"unzipping {trainingDataPath}")
      bash(["tar","zxvf",f"{extraPath}/{trainingDataPath}","-C",f"{extraPath}/{'/'.join((trainingDataPath.split('/')[:-1]))}"],shell=False)
      #os.system(f"tar zxvf {extraPath}/{'/'.join((trainingDataPath.split('/')[:-1]))} -C {extraPath}/{trainingDataPath}")
  if validationDataPath != "":
    extraPath = name
    print(f"copying {validationDataPath}")
    bashCP(f"{validationDataPath}",f"{extraPath}/validationData")
    extraPath = mimicCDOnPath(extraPath,"validationData")
    if validationDataPath[-3:] == ".gz":
      print(f"unzipping {validationDataPath}")
      try:
        bash(["gzip",f"{extraPath}/{validationDataPath}","-d",f"{extraPath}/{'/'.join((validationDataPath.split('/')[:-1]))}"],shell=False)
      except:
        pass
      validationDataPath = validationDataPath[:-3]
    if validationDataPath[-4:] == ".tar":
      print(f"unzipping {validationDataPath}")
      bash(["tar","xopf",f"{extraPath}/{validationDataPath}","--directory",f"{extraPath}/{'/'.join((validationDataPath.split('/')[:-1]))}"],shell=False)
    if validationDataPath[-4:] == ".zip":
      print(f"unzipping {validationDataPath}")
      bash(["unzip","-o","-d",f"{extraPath}/{'/'.join((validationDataPath.split('/')[:-1]))}","-qq",f"{extraPath}/{validationDataPath}"],shell=False)
    if validationDataPath[-4:] == ".tgz":
      print(f"unzipping {trainingDataPath}")
      #bash(["tar","zxvf",f"{extraPath}/{'/'.join((validationDataPath.split('/')[:-1]))}","-C",f"{extraPath}/{validationDataPath}"],shell=False)
      bash(["tar","zxvf",f"{extraPath}/{validationDataPath}","-C",f"{extraPath}/{'/'.join((validationDataPath.split('/')[:-1]))}"],shell=False)
  #%cd ../..
  print("Finished")

class Prediction:
  def __init__(self,allClasses,logPredictionValues):
    self.allClasses = allClasses
    self.logPredictionValues = logPredictionValues
    self.predictionPercents = np.exp(logPredictionValues)
    self.zipped = list(zip(self.allClasses,self.predictionPercents))
  
  def __str__(self):
    return str(self.zipped)

class Task:
	
  def removeFileEnding(x):
    #Returns everything before the first period, class function.
    return x.split('.')[0]
  
  def __init__(self,direcIn,name,rawData = "",annotations = ""):
    self.name = name                          #name of task (String)
    self.direcIn = direcIn                    #ModelDirectory
    self.rawData = rawData                    #Location of data within the trainingData and validationData folders (String)
    self.annotations = annotations            #Location of annotations within those folders (String)
    self.allDataFiles_t = []                  #List of all the training data files ([String])
    self.allAnnotationFiles_t = []            #List of all the training annotation files ([String])
    self.allDataFiles_v = []                  #List of all the validation data files ([String])
    self.allAnnotationFiles_v = []            #List of all the validation annotation files ([String])
    self.trainingDict = {}                    #IF CATEGORICAL DATA:
                                                  #Dictionaary matching name to pandas dataframe read from csv f"{name}.csv" {String:Pandas}
                                              #ELIF THERE IS ANNOTATION DATA:
                                                  #Dictionary matching training data filename to validation data filename {String:String}
                                              #ELSE: Empty {}
    self.validationDict = {}                  #Same as above, but for validation data, not training data
    self.allClasses = []                      #List of human-readable classnames, [String]
    self.fastAIData = None                    #fastAI-readable data.  Type will depend on what type of data you're using, see fastAI docs
    self.csvValidationIndices = []            #IF CALLED self.makeCSV():
                                                  #Contains a list of indices that should be used for validation rather than training [Int]
                                              #ELSE: Empty []
                                              #IF [], it is passed as None into fastAI code, which will then automatically grab a random portion of indices.
    self.fastAIModel = None                   #The actual model.  Perhabs more accurately called a PyTorch model, but it is created using fastAI
    self.curArchitecture = resnet34           #Architecture of the aforementioned model
    self.validationPredictions = None         #Predictions on the validation set, log scale.  This is currently DEPRECATED
    self.booleanPredictions = []              #Same as above, but 0 or 1 instead of log scale.  This is currently DEPRECATED
    self.predictionProbabilities = []         #Same as above, but between 0 and 1. [[Float]]
    self.allDataIsConsideredTraining = False  #IF TRUE: We let fastAI create its own validation set from the union of our training and validation sets
    self.curTransformsFromModel = None        #Data augmentation, required to predict on things not in the training of validation sets.
    self.curDataSize = 64                     #IF IMAGE DATA: curDataSize by curDataSize will be the dimensions images are scaled to.
    self.lastCSVName = None                   #The name of the last CSV created by the Task, which it will load data from when creating a model.
    self.trainingDataFrame = None             #IF TEXT DATA: A pandas dataframe consisting of training data for the model.
                                              #IF CATEGORICAL DATA: same thing but also includes validation data (indices of which are stored in 
                                                   #csvValidationIndices).  Has the dependant variable removed (the one you're predicting)
    self.validationDataFrame = None           #IF TEXT DATA: Same as above, but for validation
                                              #IF CATEGORICAL DATA: contains the dependant variable
    self.intToString = None                   #IF TEXT DATA: A dictionary mapping numbers to their corresponding tokens ({Int:String})
    self.stringToInt = None                   #IF TEXT DATA: A dictionary mapping tokens to their corresponding numbers ({String:Int})
    self.wordFrequency = None                 #IF TEXT DATA: Keeps track of the frequencies of tokens ([(String,Int)])
    self.tokenizedTraining = None             #IF TEXT DATA: Training data, tokenized.  ([[Int]])
    self.tokenizedTrainingLabels = None       #IF TEXT DATA: Training data, labeled.  ([[Int]])
    self.tokenizedValidation = None           #same
    self.tokenizedValidationLabels = None     #same
    self.typeOfData = None                    #"IMAGE" if image data, "TEXT" if text data, "CATEGORICAL" if categorical ddata
    self.preTokenizedTraining = None          #Meant for debugging - same as tokenizedTraining but not tokenized
    self.preTokenizedValidation = None        #Meant for debugging - ditto
    self.discreteVariables = None             #IF CATEGORICAL DATA: Fields that are not continuous in the dataframe
    self.continuousVariables = None           #IF CATEGORICAL DATA: Opposite ditto
            
    """TODO: Instead of recording lastCSVName, we should immediately make a dataframe of it, trainingDataFrame and validationDataFrame"""
    """Wrap makeCSV() in another function, makeDataFrame() to do both of this.  That way we keep our # of variables down and their uses consistent,"""
    """As well as getting rid of the error-prone self.lastCSVName"""
  
  def copy(self,newName,appendToDirec = True):
    #Creates an exact copy of this task with name newName,
    # adds it to the directory if appendToDirec is true,
    # and returns it.
    #This isn't really supposed to be used, it's meant so that I can update old
    # Task objects to updated code if I have to change something at runtime
    # without having to create a new one from scratch and go through all the
    # computationally expensive processes.
    # (which is often when adding features and debugging)
    toReturn = Task(self.direcIn,newName,self.rawData,self.annotations)
    toReturn.allDataFiles_t = self.allDataFiles_t
    toReturn.allAnnotationFiles_t = self.allAnnotationFiles_t
    toReturn.allDataFiles_v = self.allDataFiles_v
    toReturn.allAnnotationFiles_v = self.allAnnotationFiles_v
    toReturn.trainingDict = self.trainingDict
    toReturn.validationDict = self.validationDict
    toReturn.allClasses = self.allClasses
    toReturn.fastAIData = self.fastAIData
    toReturn.csvValidationIndices = self.csvValidationIndices
    toReturn.fastAIModel = self.fastAIModel
    toReturn.curArchitecture = self.curArchitecture
    toReturn.validationPredictions = self.validationPredictions
    toReturn.booleanPredictions = self.booleanPredictions
    toReturn.predictionProbabilities = self.predictionProbabilities
    toReturn.allDataIsConsideredTraining = self.allDataIsConsideredTraining
    toReturn.curTransformsFromModel = self.curTransformsFromModel
    toReturn.curDataSize = self.curDataSize
    toReturn.lastCSVName = self.lastCSVName
    toReturn.trainingDataFrame = self.trainingDataFrame
    toReturn.validationDataFrame = self.validationDataFrame
    toReturn.typeOfData = self.typeOfData
    toReturn.preTokenizedTraining = self.preTokenizedTraining
    toReturn.preTokenizedValidation = self.preTokenizedValidation
    toReturn.intToString = self.intToString
    toReturn.stringToInt = self.stringToInt
    toReturn.wordFrequency = self.wordFrequency
    toReturn.tokenizedTraining = self.tokenizedTraining
    toReturn.tokenizedTrainingLabels = self.tokenizedTrainingLabels
    toReturn.tokenizedValidation = self.tokenizedValidation
    toReturn.tokenizedValidationLabels = self.tokenizedValidationLabels
    toReturn.discreteVariables = self.discreteVariables
    toReturn.continuousVariables = self.continuousVariables
    if appendToDirec: self.direcIn.tasks.append(toReturn)
    return toReturn
  
  def removeDuplicateAnnotations(self):
    #This assumes your annotations are not order dependent
    #that isn't always a safe assumption, but it's unusual to want to remove
    #duplicates in ordered lists so unless problems arise, we'll stick with this.
    #Another assumption is that trainingDict/validationDict are lists
    for k, v in self.trainingDict.items():
      self.trainingDict[k] = list(set(v))
    for k, v in self.validationDict.items():
      self.validationDict[k] = list(set(v))
  
  def findClassRange(self):
    if self.fastAIModel is None:
      for k, v in self.trainingDict.items():
        self.allClasses = self.allClasses + v
      self.allClasses = list(set(self.allClasses))
      for k, v in self.validationDict.items():
        self.allClasses = self.allClasses + v
      self.allClasses = list(set(self.allClasses))
    else:
      self.allClasses = self.fastAIData.classes
    return self.allClasses
  
  def matchFiles(self,dataFunc = removeFileEnding,annotateFunc = None):
    #Links every data file with its annotation
    #Takes in 2 functions acting on the filename, the first one on the data
    #and the second on the annotations.  If no func is provided for annotations,
    #it uses the same as datafunc
    #The simplest func would be to remove file endings, so that is the default
    if annotateFunc == None:
      annotateFunc = dataFunc
    self.allDataFiles_t = bashLS(f"{self.direcIn.name}/trainingData/{self.rawData}")
    self.allAnnotationFiles_t = bashLS(f"{self.direcIn.name}/trainingData/{self.annotations}")
    self.allDataFiles_v = bashLS(f"{self.direcIn.name}/validationData/{self.rawData}")
    self.allAnnotationFiles_v = bashLS(f"{self.direcIn.name}/validationData/{self.annotations}")
	
    #Apply the match functions
    datas = list(map(lambda x: map(lambda y: (dataFunc(y),y),x),[self.allDataFiles_t,self.allDataFiles_v,]))
    annotes = list(map(lambda x: map(lambda y: (annotateFunc(y),y),x),[self.allAnnotationFiles_t,self.allAnnotationFiles_v]))
	
    #Turn them into dictionaries instead of tuples
    dataDict_t = dict((x, y) for x, y in list(datas[0]))
    dataDict_v = dict((x, y) for x, y in list(datas[1]))
    annoteDict_t = dict((x, y) for x, y in list(annotes[0]))
    annoteDict_v = dict((x, y) for x, y in list(annotes[1]))
	
    #Get all matches:
    intersects_t = set(dataDict_t.keys()) & set(annoteDict_t.keys())#set(map(lambda x: x[0],dataDict_t)) & set(map(lambda x: x[0],annoteDict_t))
    intersects_v = set(dataDict_v.keys()) & set(annoteDict_v.keys())#set(map(lambda x: x[0],dataDict_v)) & set(map(lambda x: x[0],annoteDict_v))
    self.trainingDict = dict((dataDict_t[a],annoteDict_t[a]) for a in intersects_t)
    self.validationDict = dict((dataDict_v[a],annoteDict_v[a]) for a in intersects_v)
  

  def deXMLAnnotations(self,deXMLList,grabText=True):
    #This function takes in a list of xml paths and transforms annotation data
    #into a list of just the values of the data found at those paths for each object
    #For example, test.xml = <person><name>bob</name></person>
    # yourtask.deXMLAnnotations([".//person.name"]) would transform
    # the annotation data from being stored as the string "test.xml" to being
    # stored as the list [<XMLElement: "bob">] (if grabText is true, is just ["bob"])
    print("De-annotating... (for large data sets, this may take a while)")
    for k, v in self.trainingDict.items():
      root = ET.parse(f"{self.direcIn.name}/trainingData/{self.annotations}/{v}").getroot()
      allVals = map(lambda x: root.findall(x), deXMLList)
      self.trainingDict[k] = [(item.text if grabText else item) for sublist in allVals for item in sublist]
      # ^^^ comprehension is used to flatten it out (we want 1d array, this is 2d)
    for k, v in self.validationDict.items():
      root = ET.parse(f"{self.direcIn.name}/validationData/{self.annotations}/{v}").getroot()
      allVals = map(lambda x: root.findall(x), deXMLList)
      self.validationDict[k] = [(item.text if grabText else item) for sublist in allVals for item in sublist]
    print("De-annotation complete")

  def mergeTrainingAndValidation(self):
    self.allDataIsConsideredTraining = True
  
  def makeCSV(self,csvname):
    print("Making csvs...")
    csvname = csvname[:-4] if csvname[-4:] == ".csv" else csvname
    trdata = [[f"trainingData/{self.rawData}/{k}", ' '.join(v)] for k, v in self.trainingDict.items()]
    #valdata = [] if self.allDataIsConsideredTraining else [[f"validationData/{testTask.rawData}/{k}", ' '.join(v)] for k, v in self.validationDict.items()]
    valdata = [[f"validationData/{self.rawData}/{k}", ' '.join(v)] for k, v in self.validationDict.items()]
    with open(f"{self.direcIn.name}/csvs/{csvname}_train.csv", 'w') as f:
      f.truncate()
      writer = csv.writer(f)
      rowdata = trdata
      writer.writerows(rowdata)
    with open(f"{self.direcIn.name}/csvs/{csvname}_val.csv", 'w') as f:
      f.truncate()
      writer = csv.writer(f)
      rowdata = valdata
      writer.writerows(rowdata)
    with open(f"{self.direcIn.name}/csvs/{csvname}_all.csv", 'w') as f:
      f.truncate()
      writer = csv.writer(f)
      rowdata = trdata + valdata
      writer.writerows(rowdata)
    self.csvValidationIndices = list(range(len(trdata),len(trdata) + len(valdata) - 1))
    self.lastCSVName = csvname
    print(f"Done.   3 Csvs made: {self.direcIn.name}/csvs/{csvname}_train.csv, {self.direcIn.name}/csvs/{csvname}_val.csv, and {self.direcIn.name}/csvs/{csvname}_all.csv")
  
  def makeCSVFromFolderCategories(self,csvname,folderCategories):
    #First we have to create the training and validation dictionaries
    for f in folderCategories:
      #We want to match up the f/filename with [f] (made an array b/c of multilabel classifying)
      self.allDataFiles_t = bashLS(f"{self.direcIn.name}/trainingData/{self.rawData}/{f}")
      self.allDataFiles_v = bashLS(f"{self.direcIn.name}/validationData/{self.rawData}/{f}")
      for a in self.allDataFiles_t:
        self.trainingDict[f"{f}/{a}"] = [f]
      for a in self.allDataFiles_v:
        self.validationDict[f"{f}/{a}"] = [f]
    self.makeCSV(csvname)
    self.allClasses = folderCategories
  
  def prepImageDataForFastAI(self,modelArchitecture = resnet34, dataSize = 64):
    self.typeOfData = "IMAGE"
    self.curArchitecture = modelArchitecture
    self.curDataSize = dataSize
    _, self.curTransformsFromModel = tfms_from_model(self.curArchitecture, self.curDataSize) #the uncaptured result (_) is training transforms, including flips and such
    self.fastAIData = ImageClassifierData.from_csv(f"{self.direcIn.name}", f"",f"{self.direcIn.name}/csvs/{self.lastCSVName}_all.csv",tfms=tfms_from_model(modelArchitecture, dataSize),skip_header = False, val_idxs = None if self.csvValidationIndices == [] or self.allDataIsConsideredTraining else self.csvValidationIndices)
    self.allClasses = self.fastAIData.classes
  
  def prepTextDataForFastAI(self,csvname = "textdata",chunksize = 2000,vocabularySize = 6000,minimumWordFrequency = 2):
    self.typeOfData = "TEXT"
    #prep data for pandas-ification
    trainingTexts,trainingLabels = ( list(zip(*  self.trainingDict.items()  ))  )
    validationTexts,validationLabels = ( list(zip(*  self.validationDict.items()  ))  )
	
    #convert files to the actual texts
    print("Opening text files...")
    trainingTexts = list ( map(lambda x: loadTextFile(f"{self.direcIn.name}/trainingData/{self.rawData}/{x}")  , trainingTexts)  )
    validationTexts = list ( map(lambda x: loadTextFile(f"{self.direcIn.name}/validationData/{self.rawData}/{x}")  , validationTexts)  )
    print("All text files have been successfully perused.")
	
	
    #convert labels to integers since pandas expects that
    #note that this will only work for single label classification currently
    trainingLabels = list( map(lambda x: self.allClasses.index(x[0]), trainingLabels  )   )
    validationLabels = list( map(lambda x: self.allClasses.index(x[0]), validationLabels  )   )
	
    #this is the code that would work for multilabel, but the get_all function messes up with it... :(
    #trainingLabels = list( map(lambda x: list(map(lambda y: self.allClasses.index(y),x)), trainingLabels  )   )
    #validationLabels = list( map(lambda x: list(map(lambda y: self.allClasses.index(y),x)), validationLabels  )   )
	
    #make pandas dataframes
    self.trainingDataFrame = pd.DataFrame({'text':trainingTexts,'labels':trainingLabels},columns=['labels','text'])
    self.validationDataFrame = pd.DataFrame({'text':validationTexts,'labels':validationLabels},columns=['labels','text'])
	
    #save as csv for future reference
    self.trainingDataFrame.to_csv(f"{self.direcIn.name}/csvs/{csvname}_train_pandas.csv", header=False, index=False)
    self.validationDataFrame.to_csv(f"{self.direcIn.name}/csvs/{csvname}_validation_pandas.csv", header=False, index=False)
    print(f"2 Csvs made: {self.direcIn.name}/csvs/{csvname}_train_pandas.csv and {self.direcIn.name}/csvs/{csvname}_validation_pandas.csv")
	
    #immediately load from csvs, not quite sure why I need to do this..., but I suspect its to set the chunksize
    #it doesn't take much time, so it's okay...
    self.trainingDataFrame = pd.read_csv(f"{self.direcIn.name}/csvs/{csvname}_train_pandas.csv", header=None, chunksize=chunksize)
    self.validationDataFrame = pd.read_csv(f"{self.direcIn.name}/csvs/{csvname}_validation_pandas.csv", header=None, chunksize=chunksize)
	
    #tokenize texts part 1
    print("Tokenizing...  This will take a while")
    self.preTokenizedTraining, self.tokenizedTrainingLabels = self.get_all(self.trainingDataFrame, 1)
    self.preTokenizedValidation, self.tokenizedValidationLabels = self.get_all(self.validationDataFrame, 1)
	
    #get rid of these two lines when moving to multilabel classification.
    self.tokenizedTrainingLabels = np.squeeze(self.tokenizedTrainingLabels)
    self.tokenizedValidationLabels = np.squeeze(self.tokenizedValidationLabels)
	
    #find frequencies
    self.wordFrequency = Counter(p for o in self.preTokenizedTraining for p in o)
    self.intToString = [o for o,c in self.wordFrequency.most_common(vocabularySize) if c>minimumWordFrequency]
    self.intToString.insert(0, '_pad_')
    self.intToString.insert(0, '_unk_')
    self.stringToInt = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(self.intToString)})
	
    #tokenize texts part 2
    self.tokenizedTraining = np.array([[self.stringToInt[o] for o in p] for p in self.preTokenizedTraining])
    self.tokenizedValidation = np.array([[self.stringToInt[o] for o in p] for p in self.preTokenizedValidation])
    print("Tokenizing over!  Data preparation complete.")
  
  def setType(self,typename):
    self.typeOfData = typename
            
  def prepCategoricalDataForFastAI(self,categoryToBreakOff):
    self.typeOfData = "CATEGORICAL"
    if self.validationDict["merged"] is not None:
      self.trainingDataFrame = pd.concat([self.validationDict["merged"],self.trainingDict["merged"]], axis=1, join='inner')
      self.csvValidationIndices = range(0,len(self.validationDict["merged"]))
    else:
      self.trainingDataFrame = self.trainingDict["merged"]
      self.csvValidationIndices = random.choice(range(0,len(self.trainingDict["merged"])))[:len(self.trainingDict["merged"]) // 5]
    for v in self.discreteVariables: self.trainingDataFrame[v] = self.trainingDataFrame[v].apply(lambda x: x.astype('category').cat.as_ordered())
    for v in self.continuousVariables: self.trainingDataFrame[v] = self.trainingDataFrame[v].apply(lambda x: x.fillna(0).astype('float32'))
    for v in self.trainingDataFrame:
      if not is_numeric_dtype(type(v)): print(f"{v}:{type(v)}")
    self.trainingDataFrame, self.validationDataFrame, _, self.curTransformsFromModel = proc_df(self.trainingDataFrame, categoryToBreakOff, do_scale=True)

  def createModel(self):
    if self.typeOfData == "IMAGE":
      #Creating an image model is easy
      self.fastAIModel = ConvLearner.pretrained(self.curArchitecture, self.fastAIData)
    elif self.typeOfData == "TEXT":
      #Creating a text model is ugly
      bptt,em_sz,nh,nl = 70,400,1150,3
      vs = len(self.intToString)
      opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
      bs = 48
      min_lbl = self.tokenizedTrainingLabels.min()
      self.tokenizedTrainingLabels -= min_lbl
      self.tokenizedValidationLabels -= min_lbl
      c=int(self.tokenizedTrainingLabels.max())+1
      trainingTextDataset = TextDataset(self.tokenizedTraining, self.tokenizedTrainingLabels)
      validationTextDataset = TextDataset(self.tokenizedValidation, self.tokenizedValidationLabels)
      trn_samp = SortishSampler(self.tokenizedTraining, key=lambda x: len(self.tokenizedTraining[x]), bs=bs//2)
      val_samp = SortSampler(self.tokenizedValidation, key=lambda x: len(self.tokenizedValidation[x]))
      trainingDataLoader = DataLoader(trainingTextDataset, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
      validationDataLoader = DataLoader(validationTextDataset, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
      md = ModelData(f"{self.direcIn.name}/models", trainingDataLoader, validationDataLoader)
      dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5#dropouts
      m = lm_rnn.get_rnn_classifer(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
          layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
          dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
      opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
      self.fastAIModel = RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
      self.fastAIModel.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
      self.fastAIModel.clip=25.
      self.fastAIModel.metrics = [accuracy]
    elif self.typeOfData == "CATEGORICAL":
      self.fastAIModel = ColumnarModelData.from_data_frame(f"{self.direcIn.name}/misc", self.csvValidationIndices, self.trainingDataFrame, np.log(self.validationDataFrame).astype(np.float32), cat_flds=self.continuousVariables, bs=128)
    else:
      print("Error - Task has no type.  You can fix this with Task.setType('IMAGE' or 'TEXT')")
  
		  

  def trainModel(self,learningRate = 0.2, epochs = 1, startingCycleEpochLength = 1, subsequentCycleEpochLengthMultiplier = 1):
    print("Starting training...  This may take a long time.")
    self.fastAIModel.fit(learningRate, epochs, cycle_len=startingCycleEpochLength, cycle_mult=subsequentCycleEpochLengthMultiplier)
    print("Training over.")
    print("Storing predictions...")
    self.validationPredictions = self.fastAIModel.predict()
    self.booleanPredictions = np.argmax(self.validationPredictions, axis=1)  # from log probabilities to 0 or 1
    #^^^ NOTE TO SELF, this only works for 1 category currently due to argmax
    self.predictionProbabilities = np.exp(self.validationPredictions[:,1])
    print("Predictions stored.")
  
  #predicts given an index in the validation set
  def predictFromIndex(self,n):
    #return list(zip(self.allClasses,np.exp(self.fastAIModel.predict_array(self.fastAIData.val_ds[n][0][None])[0])))
    return Prediction(self.allClasses,self.fastAIModel.predict_array(self.fastAIData.val_ds[n][0][None])[0])

  #predicts given an array of raw data
  #Warning - the array is unlikely to be scaled properly if you pass it in, because there is preprocessing that fastAI does
  #If doing image processing, you are much better off using predictFromImageFile
  def predictFromArray(self,arrayToPredictFrom):
    #return list(zip(self.allClasses,np.exp(self.fastAIModel.predict_array(arrayToPredictFrom[None])[0])))
    return Prediction(self.allClasses,self.fastAIModel.predict_array(arrayToPredictFrom[None])[0])
  
  def predictFromText(self, text):
    #fastAI has a function predict_text which fails because it does not put things to cuda
    #this is that function but it does put things on cuda
    #for some reason it always returns the same values, so there's another bug somewhere probably
    #:(
  
    # prefix text with tokens:
    #   xbos: beginning of sentence
    #   xfld 1: we are using a single field here
    input_str = 'xbos xfld 1 ' + text
    # predictions are done on arrays of input.
    # We only have a single input, so turn it into a 1x1 array
    texts = [input_str]
    # tokenize using the fastai wrapper around spacy
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    # turn into integers for each word
    encoded = [self.stringToInt[p] for p in tok[0]]
    # we want a [x,1] array where x is the number
    #  of words inputted (including the prefix tokens)
    ary = np.reshape(np.array(encoded),(-1,1))
    # turn this array into a tensor
    tensor = torch.from_numpy(ary)
    # wrap in a torch Variable
    variable = Variable(tensor)
    variable = variable.cuda()
    # do the predictions
    self.fastAIModel.model.cuda()#just incase, idk if necessary
    predictions = self.fastAIModel.model(variable)
    # convert back to numpy
    numpy_preds = predictions[0].data.cpu().numpy()
    return numpy_preds#softmax(numpy_preds[0])[0]

  #Given an absolute file name, it will predict the output
  def predictFromImageFile(self,fileName): return self.predictFromArray(self.curTransformsFromModel(open_image(fileName)))

  def saveModel(self):
    self.fastAIModel.save(f"{self.direcIn.name}/models/{self.name}")

  def loadModel(self):
    self.fastAIModel = ConvLearner.pretrained(self.curArchitecture, self.fastAIData).load(f"{self.direcIn.name}/models/{self.name}")

  def loadTrainingCSVsAsPandas(self,csvs):
    self.trainingDict = {fname: pd.read_csv(f"{self.direcIn.name}/trainingData/{self.rawData}/{fname}.csv", low_memory = False) for fname in csvs}
            
  def loadValidationCSVsAsPandas(self,csvs):
    self.validationDict = {fname: pd.read_csv(f"{self.direcIn.name}/trainingData/{self.rawData}/{fname}.csv", low_memory = False) for fname in csvs}
            
  def peekAtCategoricalData(self,which = [],training=True,size=5):
    if not which:#if empty
      which = [k for k,v in (self.trainingDict.items() if training else self.validationDict.items())]
    if training: 
      for t in which: display(self.trainingDict[t].head(size))
    else: 
      for t in which: display(self.validationDict[t].head(size))
            
  def deepPeekAtCategoricalData(self,which,training=True):
    print("Due to a bug or update in pandas, this no longer works")
    if not which:#if empty
      which = [k for k,v in (self.trainingDict.items() if training else self.validationDict.items())]
    if training: 
      for t in which: display(DataFrameSummary(self.trainingDict[t]).summary())
    else: 
      for t in which: display(DataFrameSummary(self.validationDict[t]).summary())
            
  def getAllCategoricalColumns(self):
    toReturn = []
    for k,v in self.trainingDict.items():
      toReturn = toReturn + v.columns.values.tolist()
    return list(set(toReturn))
            
  def editCategoricalData(self,tableName,columnName,funcToPerform,newcol=None,training=True):
    reldf = None
    if training:
      reldf = self.trainingDict[tableName]
    else:
      reldf = self.validationDict[tableName]
    reldf[columnName if newcol is None else newcol] = reldf[columnName].apply(funcToPerform)
            
  def mergeCategoricalDataFrames(self):
    trlist = [v for k,v in self.trainingDict.items()]
    self.trainingDict = {"merged":pd.concat(trlist, axis=1, join='inner')}
    vallist = [v for k,v in self.validationDict.items()]
    self.validationDict = {"merged":pd.concat(trlist, axis=1, join='inner')} 
            
  def expandCategoricalDates(self,nameOfDateColumn):
    for d in [self.trainingDict,self.validationDict]:
      for k,v in d.items():
        if v[nameOfDateColumn] is None: continue
        add_datepart(v, nameOfDateColumn, drop=False)
    
  def setCategoricalVariables(self,discrete,continuous):
    self.continuousVariables = continuous
    self.discreteVariables = discrete
    if self.trainingDict["merged"] is None:
      print("ERROR: Must be called after mergeCategoricalDataFrames")
      return
    self.trainingDict["merged"] = self.trainingDict["merged"][continuous+discrete]
    for col in continuous:
      self.trainingDict["merged"][col].fillna(0).astype('float32')
    if self.validationDict["merged"] is None: return
    self.validationDict["merged"] = self.validationDict["merged"][continuous+discrete]
    for col in continuous:
      self.validationDict["merged"][col].fillna(0).astype('float32')
            
  #The grab random display code is taken pretty much verbatim from here, although names changed to fit with theme
  #https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb
  #def randomByMask(self,mask,amount = 4): return np.random.choice(np.where(mask)[0], amount, replace=False)
  #def getRandomValidationFromCorrectness(self,is_correct=True,amount = 4): return self.randomByMask((self.booleanPredictions == self.fastAIData.val_y)==is_correct, amount)
  
  def plots(self,ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
  
  def load_img_id(self,ds, idx): return np.array(PIL.Image.open(f"{self.direcIn.name}/{ds.fnames[idx]}"))

  def drawRandomPlots(self, amount = 4):
      idxs = np.random.choice(len(self.fastAIData.val_ds)-1,amount,replace=False)
      return self.drawPlots(idxs)
  
  def drawPlots(self, idxs):
      imgs = [self.load_img_id(self.fastAIData.val_ds,x) for x in idxs]
      title_probs = [f"{self.fastAIData.val_ds.fnames[x].split('/')[-1]} :{self.predictionProbabilities[x]}" for x in idxs]
      return self.plots(imgs, rows=1, titles=title_probs, figsize=(16,8))

  #Next three functions taken directly from link below, meant for loading data from a pandas dataframe correctly
  #https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
  def fixup(self,x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

  def get_texts(self,df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\nxbos xfld 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' xfld {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(self.fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)
  
  def get_all(self,df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = self.get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

class ModelDirectory:
	
  def copy(self):
    #This is used for debuggings and whatnots
    toReturn = ModelDirectory(self.name,self.trainingDataPath,self.validationDataPath,False)
    toReturn.tasks = []
    return toReturn
  
  def __init__(self,name,trainingDataPath = "",validationDataPath = "",create=True):
    self.name = name
    self.trainingDataPath = trainingDataPath
    self.validationDataPath = validationDataPath
    self.tasks = []
    if create: createModelDirectory(name,trainingDataPath,validationDataPath)
  
  def giveTask(self,taskname,rawData = "",annotations = "",folderCategories = [],trainingCSVNames = [],validationCSVNames = []):
    """
      taskname: ID given to task
      rawData: filepath to training/validation data location, assuming root directory is self.direcIn.name/trainingData and self.direcIn.name/validationData
      ^^^ The above two are basically always needed, although if you're lucky rawData can just be ""
      folderCategories: If you have files in folders that denote the label of the file, use this.  Works for text or image data.
      trainingCSVNames/validationCSVNames: Names of csvs that contain the categorical data.  At the moment it is mutually exclusive with folderCategories
        and is only meant to be used for categorical data
    """
    newTask = Task(self,taskname,rawData,annotations)
    if folderCategories: #if folderCategories is not empty
      newTask.makeCSVFromFolderCategories(self.name,folderCategories)
    if trainingCSVNames:
      newTask.loadTrainingCSVsAsPandas(trainingCSVNames)
      newTask.loadValidationCSVsAsPandas(validationCSVNames)
    self.tasks.append(newTask)
    return newTask
  
  def getTask(self,taskname):
    for a in self.tasks:
      if a.name == taskname:
        return a
    return None
  
  def extractJumbledValidation(self,foldersToPrepend,fileLocation):
    print("Extracting...")
    curdirec = f"{self.name}/validationData"
    stringToAppend = ""
    if isinstance(foldersToPrepend,str):#if passed in "a/string/like/this"
      stringToAppend = foldersToPrepend
      foldersToPrepend = stringToAppend.split("/")
    else:#if passed in ["a","list","of","nested","directories"]
      stringToAppend = "/".join(foldersToPrepend)
    for folder in foldersToPrepend:
      makeDirectory(f"{curdirec}/{folder}")
      curdirec = mimicCDOnPath(curdirec,folder)
    bashCP(f"{self.name}/trainingData/{fileLocation}",f"{self.name}/validationData/{stringToAppend}")
    print("Extracting over.")
                     
                     

