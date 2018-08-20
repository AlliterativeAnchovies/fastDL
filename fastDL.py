
import xml.etree.ElementTree as ET
import csv
from fastai.text import *
from fastai.dataset import *
from fastai.conv_learner import *
from fastai import lm_rnn
import matplotlib.pyplot as plt
import PIL
import spacy
import html
import subprocess
import os

spacy.load("en")
re1 = re.compile(r'  +')

def loadTextFile(fileName):
  toReturn = Path(fileName).open('r').read()#!cat {fileName} >/dev/null
  return toReturn

def bashLS(directory):
  return subprocess.check_output(["ls","-1",directory],shell=True).decode("utf-8").split("\n")[:-1]

def bashCP(cpFrom,cpTo):
  subprocess.check_output(["cp","-r",cpFrom,cpTo],shell=True)

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
    bashCP(f"{extraPath}/../{trainingDataPath}","trainingData")
    extraPath = mimicCDOnPath(extraPath,"trainingData")
    if trainingDataPath[-3:] == ".gz":
      print(f"unzipping {trainingDataPath}")
      !gzip -d {extraPath}/{trainingDataPath}
      trainingDataPath = trainingDataPath[:-3]
    if trainingDataPath[-4:] == ".tar":
      print(f"unzipping {trainingDataPath}")
      !tar xopf {extraPath}/{trainingDataPath}
    if trainingDataPath[-4:] == ".zip":
      print(f"unzipping {trainingDataPath}")
      !unzip -qq {extraPath}/{trainingDataPath}
  if validationDataPath != "":
    extraPath = mimicCDOnPath(extraPath,"..")
    print(f"copying {validationDataPath}")
    bashCP(f"{extraPath}/../{validationDataPath}","validationData")
    extraPath = mimicCDOnPath(extraPath,"validationData")
    if validationDataPath[-3:] == ".gz":
      print(f"unzipping {validationDataPath}")
      !gzip -d {extraPath}/{validationDataPath}
      validationDataPath = validationDataPath[:-3]
    if validationDataPath[-4:] == ".tar":
      print(f"unzipping {validationDataPath}")
      !tar xopf {extraPath}/{validationDataPath}
    if validationDataPath[-4:] == ".zip":
      print(f"unzipping {validationDataPath}")
      !unzip -qq {extraPath}/{validationDataPath}
  #%cd ../..
  print("Finished")

class Task:
	
  def removeFileEnding(x):
    #Returns everything before the first period, class function.
    return x.split('.')[0]
  
  def __init__(self,direcIn,name,rawData = "",annotations = ""):
    self.name = name
    self.direcIn = direcIn
    self.rawData = rawData
    self.annotations = annotations
    self.allDataFiles_t = []
    self.allAnnotationFiles_t = []
    self.allDataFiles_v = []
    self.allAnnotationFiles_v = []
    self.trainingDict = {}
    self.validationDict = {}
    self.allClasses = []
    self.fastAIData = None
    self.csvValidationIndices = []
    self.fastAIModel = None
    self.curArchitecture = resnet34
    self.validationPredictions = None
    self.booleanPredictions = []
    self.predictionProbabilities = []
    self.allDataIsConsideredTraining = True
    self.curTransformsFromModel = None
    self.curDataSize = 64
    self.lastCSVName = None
    self.tokenizerData = None
    self.trainingDataFrame = None
    self.validationDataFrame = None
    self.intToString = None
    self.stringToInt = None
    self.wordFrequency = None
    self.tokenizedTraining = None
    self.tokenizedTrainingLabels = None
    self.tokenizedValidation = None
    self.tokenizedValidationLabels = None
    self.typeOfData = None
    self.preTokenizedTraining = None
    self.preTokenizedValidation = None
  
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
    toReturn.tokenizerData = self.tokenizerData
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
    #self.trainingDict.update(self.validationDict)
    #self.validationDict = []
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
    trainingTexts,trainingLabels = ( list(zip(*  imdbTempTask.trainingDict.items()  ))  )
    validationTexts,validationLabels = ( list(zip(*  imdbTempTask.validationDict.items()  ))  )
	
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
  def predictFromIndex(self,n): return self.fastAIModel.predict_array(self.fastAIData.val_ds[n][0][None])

  #predicts given an array of raw data
  #Warning - the array is unlikely to be scaled properly if you pass it in, because there is preprocessing that fastAI does
  #If doing image processing, you are much better off using predictFromImageFile
  def predictFromArray(self,arrayToPredictFrom): return self.fastAIModel.predict_array(arrayToPredictFrom[None])

  #Given an absolute file name, it will predict the output
  def predictFromImageFile(self,fileName): return self.predictFromArray(self.curTransformsFromModel(open_image(fileName)))
  
  def saveModel():
    self.fastAIModel.save(f"{self.direcIn.name}/models/{self.name}")
  
  def loadModel():
    self.fastAIModel = ConvLearner.pretrained(self.curArchitecture, self.fastAIData).load(f"{self.direcIn.name}/models/{self.name}")
  
  
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
  
  def giveTask(self,taskname,rawData = "",annotations = "",folderCategories = []):
    newTask = Task(self,taskname,rawData,annotations)
    if folderCategories: #if folderCategories is not empty
      newTask.makeCSVFromFolderCategories(self.name,folderCategories)
    self.tasks.append(newTask)
    return newTask
  
  def getTask(self,taskname):
    for a in self.tasks:
      if a.name == taskname:
        return a
    return None
  
  def extractJumbledValidation(self,foldersToPrepend,fileLocation):
    curdirec = !pwd
    curdirec = curdirec[0]
    %cd {self.name}/validationData
    stringToAppend = "/".join(foldersToPrepend)
    for folder in foldersToPrepend:
      makeDirectory(folder)
      %cd {folder}
    %cd /{curdirec}/{self.name}
    bashCP(f"trainingData/{fileLocation}",f"validationData/{stringToAppend}")
    %cd /{curdirec}
                     
