# 2019/01/10~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu
"""
train.py Training Module

Methods for training the models.

MultipleModels: Handles the training for multiple models simultaneously
"""

import torch
import numpy as np
import os
import pickle
import datetime

def MultipleModels(modelsDict, data, nEpochs, batchSize,
        **kwargs):
    """
    Trains multiple models simultaneously

    Inputs:

        modelsDict (dict): Dictionary containing the models to be trained (see
            Modules.model.Model class)
        data (class): Data to carry out the training (see Utils.dataTools)
        nEpochs (int): number of epochs (passes over the dataset)
        batchSize (int): size of each minibatch

        Keyword arguments:

        validationInterval (int): interval of training (number of training
            steps) without running a validation stage.

        Optional (keyword) arguments:

        learningRateDecayRate (float): float that multiplies the latest learning
            rate used.
        learningRateDecayPeriod (int): how many training steps before 
            multiplying the learning rate decay rate by the actual learning
            rate.
        > Obs.: Both of these have to be defined for the learningRateDecay
              scheduler to be activated.
        logger (Visualizer): save tensorboard logs.
        saveDir (string): path to the directory where to save relevant training
            variables.
        printInterval (int): how many training steps after which to print
            partial results (0 means do not print)
        graphNo (int): keep track of what graph realization this is
        realitizationNo (int): keep track of what data realization this is
        >> Alternatively, these last two keyword arguments can be used to keep
            track of different trainings of the same model

    Observations:
    - Model parameters for best and last are saved.

    """

    ####################################
    # ARGUMENTS (Store chosen options) #
    ####################################

    # Training Options:
    if 'logger' in kwargs.keys():
        doLogging = True
        logger = kwargs['logger']
    else:
        doLogging = False

    if 'saveDir' in kwargs.keys():
        doSaveVars = True
        saveDir = os.path.join(kwargs['saveDir'],'trainVars')
    else:
        doSaveVars = False

    if 'printInterval' in kwargs.keys():
        doPrint = True
        printInterval = kwargs['printInterval']
    else:
        doPrint = False

    if 'learningRateDecayRate' in kwargs.keys() and \
        'learningRateDecayPeriod' in kwargs.keys():
        doLearningRateDecay = True
        learningRateDecayRate = kwargs['learningRateDecayRate']
        learningRateDecayPeriod = kwargs['learningRateDecayPeriod']
    else:
        doLearningRateDecay = False

    validationInterval = kwargs['validationInterval']

    if 'graphNo' in kwargs.keys():
        graphNo = kwargs['graphNo']

    if 'realizationNo' in kwargs.keys():
        realizationNo = kwargs['realizationNo']

    # No training case:
    if nEpochs == 0:
        doSaveVars = False
        doLogging = False
        # If there's no training happening, there's nothing to report about
        # training losses and stuff.
            
    # Get the device we're working on
    device = None # Not set
    for key in modelsDict.keys():
        params = list(modelsDict[key].archit.parameters())
        thisDevice = params[0].device
        if device is None:
            device = thisDevice
        else:
            assert device == thisDevice


    ###########################################
    # DATA INPUT (pick up on data parameters) #
    ###########################################

    nTrain = len(data) # size of the training set

    # Number of batches: If the desired number of batches does not split the
    # dataset evenly, we reduce the size of the last batch (the number of
    # samples in the last batch).
    # The variable batchSize is a list of length nBatches (number of batches),
    # where each element of the list is a number indicating the size of the
    # corresponding batch.
    if nTrain < batchSize:
        nBatches = 1
        batchSize = [nTrain]
    elif nTrain % batchSize != 0:
        nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
        batchSize = [batchSize] * nBatches
        # If the sum of all batches so far is not the total number of graphs,
        # start taking away samples from the last batch (remember that we used
        # ceiling, so we are overshooting with the estimated number of batches)
        while sum(batchSize) != nTrain:
            batchSize[-1] -= 1
    # If they fit evenly, then just do so.
    else:
        nBatches = np.int(nTrain/batchSize)
        batchSize = [batchSize] * nBatches
    # batchIndex is used to determine the first and last element of each batch.
    # If batchSize is, for example [20,20,20] meaning that there are three
    # batches of size 20 each, then cumsum will give [20,40,60] which determines
    # the last index of each batch: up to 20, from 20 to 40, and from 40 to 60.
    # We add the 0 at the beginning so that batchIndex[b]:batchIndex[b+1] gives
    # the right samples for batch b.
    batchIndex = np.cumsum(batchSize).tolist()
    batchIndex = [0] + batchIndex

    ##############
    # TRAINING   #
    ##############

    # Learning rate scheduler:
    if doLearningRateDecay:
        learningRateScheduler = {}
        for key in modelsDict.keys():
            learningRateScheduler[key] = torch.optim.lr_scheduler.StepLR(
                    modelsDict[key].optim, learningRateDecayPeriod,
                    learningRateDecayRate)

    # Initialize counters (since we give the possibility of early stopping, we
    # had to drop the 'for' and use a 'while' instead):
    epoch = 0 # epoch counter

    #\\\ Save variables to be used for each model
    # Logging variables
    if doLogging:
        lossTrainTB = {}
        evalTrainTB = {}
        lossValidTB = {}
        evalValidTB = {}

    # Training variables of interest (to save in these dictionaries).
    #   The key of each dictionary is the model name.
    if doSaveVars:
        lossTrain = {}
        evalTrain = {}
        lossValid = {}
        evalValid = {}
        timeTrain = {}
        timeValid = {}
        for key in modelsDict.keys():
            lossTrain[key] = []
            evalTrain[key] = []
            lossValid[key] = []
            evalValid[key] = []
            timeTrain[key] = []
            timeValid[key] = []

    # Model tracking
    bestScore = {}
    bestEpoch = {}
    bestBatch = {}

    for epoch in range(nEpochs):

        # Randomize dataset for each epoch
        randomPermutation = np.random.permutation(nTrain)
        # Convert a numpy.array of numpy.int into a list of actual int.
        idxEpoch = [int(i) for i in randomPermutation]

        # Learning decay
        if doLearningRateDecay:
            for key in learningRateScheduler.keys():
                learningRateScheduler[key].step()

            if doPrint:
                # All the optimization have the same learning rate, so just
                # print one of them
                # TODO: Actually, they might be different, so I will need to
                # print all of them.
                print("Epoch %d, learning rate = %.8f" % (epoch+1,
                      learningRateScheduler[key].optim.param_groups[0]['lr']))

        # Initialize counter

        for batch in range(nBatches - 1):

            # Extract the adequate batch
            thisBatchIndices = idxEpoch[batchIndex[batch] : batchIndex[batch+1]]
            # Get the samples
            xTrain, STrain, yTrain = data.__getitem__(thisBatchIndices)
            # Move samples to device
            xTrain = np.transpose(xTrain, axes = [0,2,1])

            xTrain = torch.tensor(xTrain)
            STrain = torch.tensor(STrain)
            STrain = STrain.unsqueeze(1)


            yTrain = torch.tensor(yTrain)


            xTrain = xTrain.to(device)
            yTrain = yTrain.to(device)
            STrain = STrain.to(device)

            # Print if doPrint is selected and every printInterval iterations
            if doPrint and printInterval > 0:
                if (epoch * nBatches + batch) % printInterval == 0:
                    trainPreamble = ''
                    if 'graphNo' in kwargs.keys():
                        trainPreamble += 'G:%02d ' % graphNo
                    if 'realizationNo' in kwargs.keys():
                        trainPreamble += 'R:%02d ' % realizationNo
                    print("[%sTRAINING - E: %2d, B: %3d]" % (
                            trainPreamble, epoch+1, batch+1))

            for key in modelsDict.keys():
                
                # Start measuring time
                startTime = datetime.datetime.now()

                # Reset gradients
                modelsDict[key].archit.zero_grad()

                # Obtain the output of the GNN
                # If we are in local GNN where we want to compute the output
                # at a single node, then we need to use that, instead of the
                # conventional forward
                if 'singleNodeForward' in dir(modelsDict[key].archit):
                    # This, entails getting at which nodes we want to compute
                    # the output
                    if 'getLabelID' in dir(data):
                        targetIDs = data.getLabelID('train', thisBatchIndices)
                        yHatTrain = modelsDict[key].archit.singleNodeForward(
                                                                     xTrain,
                                                                     targetIDs)
                else:
                    yHatTrain = modelsDict[key].archit(xTrain, STrain)

                # Compute loss
                lossValueTrain = modelsDict[key].loss(yHatTrain, yTrain)

                # Compute gradients
                lossValueTrain.backward()

                # Optimize
                modelsDict[key].optim.step()
                
                # Finish measuring time
                endTime = datetime.datetime.now()
                
                timeElapsed = abs(endTime - startTime).total_seconds()

                # Compute the accuracy
                #   Note: Using yHatTrain.data creates a new tensor with the
                #   same value, but detaches it from the gradient, so that no
                #   gradient operation is taken into account here.
                #   (Alternatively, we could use a with torch.no_grad():)
                dataType = type(yTrain)

                N = len(yTrain)
        
                yTrain = torch.squeeze(yTrain)
        
                if 'torch' in repr(dataType):
            #   We compute the target label (hardmax)
                    yHatTrain.data = torch.argmax(yHatTrain.data, dim = 1)
            #   And compute the error
            # print(yHatTrain.data-yTrain)
                    totalErrors = torch.sum(torch.abs(yHatTrain.data - yTrain) > 1e-9)
            # print(totalErrors.type(dataType))
                    errorRate = totalErrors.type(dataType)/N
            # print(errorRate)
                else:
                    yHatTrain.data = np.array(yHatTrain.data)
                    yTrain = np.array(yTrain)
            #   We compute the target label (hardmax)
                    yHatTrain.data = np.argmax(yHatTrain.data, axis = 1)
            #   And compute the error
                    totalErrors = np.sum(np.abs(yHatTrain.data - yTrain) > 1e-9)
                    errorRate = totalErrors.astype(dataType)/N

                accTrain = errorRate

                # Logging values
                if doLogging:
                    lossTrainTB[key] = lossValueTrain.item()
                    evalTrainTB[key] = accTrain.item()

                # Save values
                if doSaveVars:
                    lossTrain[key] += [lossValueTrain.item()]
                    evalTrain[key] += [accTrain.item()]
                    timeTrain[key] += [timeElapsed]                    

                # Print:
                if doPrint and printInterval > 0:
                    if (epoch * nBatches + batch) % printInterval == 0:
                        print("\t(%s) %6.4f / %7.4f - %6.4fs" % (
                                    key, accTrain.item(), lossValueTrain.item(),
                                    timeElapsed))
                        
                # Get rid of intermediate variables to free, if necessary,
                # CUDA space (Note that using .item() already detaches the
                # scalar value and converts it into a float on CPU)
                del yHatTrain
                del lossValueTrain
                del accTrain

            #\\\\\\\
            #\\\ TB LOGGING (for each batch)
            #\\\\\\\

            if doLogging:
                modeLoss = 'Loss'
                modeEval = 'Accuracy'
                if 'graphNo' in kwargs.keys():
                    modeLoss += 'G%02d' % graphNo
                    modeEval += 'G%02d' % graphNo
                if 'realizationNo' in kwargs.keys():
                    modeLoss += 'R%02d' % realizationNo
                    modeEval += 'R%02d' % realizationNo
                logger.scalar_summary(mode = 'Training' + modeLoss,
                                      epoch = epoch * nBatches + batch,
                                      **lossTrainTB)
                logger.scalar_summary(mode = 'Training' + modeEval,
                                      epoch = epoch * nBatches + batch,
                                      **evalTrainTB)

            #\\\\\\\
            #\\\ VALIDATION
            #\\\\\\\

            if (epoch * nBatches + batch) % validationInterval == 0:
                # Validation:
                xValid, SValid, yValid = data.__getitem__(idxEpoch[batchIndex[-3]: batchIndex[-1]])
                # Move to device
                SValid  = torch.tensor(SValid )
                SValid = SValid.unsqueeze(1)
                xValid = np.transpose(xValid, axes = [0,2,1])

                xValid = torch.tensor(xValid)
                yValid = torch.tensor(yValid)

                xValid = xValid.to(device)
                yValid = yValid.to(device)
                SValid = SValid.to(device)

                if doPrint:
                    validPreamble = ''
                    if 'graphNo' in kwargs.keys():
                        validPreamble += 'G:%02d ' % graphNo
                    if 'realizationNo' in kwargs.keys():
                        validPreamble += 'R:%02d ' % realizationNo
                    print("[%sVALIDATION - E: %2d, B: %3d]" % (
                            validPreamble, epoch+1, batch+1))

                for key in modelsDict.keys():
                    
                    # Start measuring time
                    startTime = datetime.datetime.now()
                    
                    # Under torch.no_grad() so that the computations carried out
                    # to obtain the validation accuracy are not taken into
                    # account to update the learnable parameters.
                    with torch.no_grad():
                        # Obtain the output of the GNN
                        if 'singleNodeForward' in dir(modelsDict[key].archit):
                            if 'getLabelID' in dir(data):
                                targetIDs = data.getLabelID('valid')
                                yHatValid = modelsDict[key].archit\
                                            .singleNodeForward(xValid,targetIDs)
                        else:
                            yHatValid = modelsDict[key].archit(xValid, SValid)

                        # Compute loss
                        lossValueValid = modelsDict[key].loss(yHatValid, yValid)
                                       
                        # Finish measuring time
                        endTime = datetime.datetime.now()
                        
                        timeElapsed = abs(endTime - startTime).total_seconds()

                        # Compute accuracy:
                        dataType = type(yValid)

                        N = len(yValid)
                        yValid = torch.squeeze(yValid)

                        if 'torch' in repr(dataType):
            #   We compute the target label (hardmax)
                            yHatValid.data = torch.argmax(yHatValid.data, dim = 1)
            #   And compute the error
                            totalErrors = torch.sum(torch.abs(yHatValid.data - yValid) > 1e-9)
                            errorRate = totalErrors.type(dataType) / N
                        else:
                            yHatValid.data = np.array(yHatValid.data)
                            yValid = np.array(yValid)
            #   We compute the target label (hardmax)
                            yHatValid.data = np.argmax(yHatValid.data, axis = 1)
            #   And compute the error
                            totalErrors = np.sum(np.abs(yHatValid.data - yValid) > 1e-9)
                            errorRate = totalErrors.astype(dataType) / N

                        accValid = errorRate

                        # Logging values
                        if doLogging:
                            lossValidTB[key] = lossValueValid.item()
                            evalValidTB[key] = accValid.item()

                        # Save values
                        if doSaveVars:
                            lossValid[key] += [lossValueValid.item()]
                            evalValid[key] += [accValid.item()]
                            timeValid[key] += [timeElapsed]

                        # Print:
                        if doPrint:
                            print("\t(%s) %6.4f / %7.4f - %6.4fs" % (
                                    key, accValid, lossValueValid.item(),
                                    timeElapsed))

                    # No previous best option, so let's record the first trial
                    # as the best option
                    if epoch == 0 and batch == 0:
                        bestScore[key] = accValid
                        bestEpoch[key], bestBatch[key] = epoch, batch
                        # Save this model as the best (so far)
                        modelsDict[key].save(label = 'Best')
                        # Store the keys of the best models when they happen
                        keyBest = []
                    else:
                        thisValidScore = accValid
                        if thisValidScore < bestScore[key]:
                            bestScore[key] = thisValidScore
                            bestEpoch[key], bestBatch[key] = epoch, batch
                            if doPrint:
                                keyBest += [key]
                            modelsDict[key].save(label = 'Best')
                            
                    # Get rid of intermediate variables to free, if necessary,
                    # CUDA space (Note that using .item() already detaches the
                    # scalar value and converts it into a float on CPU)
                    del yHatValid
                    #   These ones coming up next, are all scalar, so it would
                    #   not make much of a difference to delete them now, but
                    #   whatever (they might have gradients that could occupy
                    #   a lot of space)
                    del lossValueValid
                    del accValid

                if doPrint:
                    if len(keyBest) > 0:
                        for key in keyBest:
                            print("\t=> New best achieved for %s: %.4f" % \
                                              (key, bestScore[key]))
                        keyBest = []

                if doLogging:
                    logger.scalar_summary(mode = 'Validation' + modeLoss,
                                          epoch = epoch * nBatches + batch,
                                          **lossValidTB)
                    logger.scalar_summary(mode = 'Validation' + modeEval,
                                          epoch = epoch * nBatches + batch,
                                          **evalValidTB)

            #\\\\\\\
            #\\\ END OF BATCH
            #\\\\\\\

        #\\\\\\\
        #\\\ END OF EPOCH
        #\\\\\\\

    #\\\ Save models:
    for key in modelsDict.keys():
        modelsDict[key].save(label = 'Last')

    #################
    # TRAINING OVER #
    #################

    if doSaveVars:
        # We convert the lists into np.arrays to be handled by matplotlib
        for key in modelsDict.keys():
            lossTrain[key] = np.array(lossTrain[key])
            evalTrain[key] = np.array(evalTrain[key])
            timeTrain[key] = np.array(timeTrain[key])
            lossValid[key] = np.array(lossValid[key])
            evalValid[key] = np.array(evalValid[key])
            timeValid[key] = np.array(timeValid[key])
        # And we would like to save all the relevant information from training
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        # Dictionaries of variables to save
        varsPickle = {}
        # And save the relevant variables
        varsPickle['nEpochs'] = nEpochs
        varsPickle['nBatches'] = nBatches
        varsPickle['validationInterval'] = nBatches
        varsPickle['batchSize'] = np.array(batchSize)
        varsPickle['batchIndex'] = np.array(batchIndex)
        varsPickle['bestBatch'] = bestBatch
        varsPickle['bestEpoch'] = bestEpoch
        varsPickle['bestScore'] = bestScore
        varsPickle['lossTrain'] = lossTrain
        varsPickle['evalTrain'] = evalTrain
        varsPickle['timeTrain'] = timeTrain
        varsPickle['lossValid'] = lossValid
        varsPickle['evalValid'] = evalValid
        varsPickle['timeValid'] = timeValid
        # Create file for pickling
        varsFilename = 'trainVars'
        # And add the information if this is a specific realization run
        if 'graphNo' in kwargs.keys():
            varsFilename += 'G%02d' % graphNo
            varsPickle['graphNo'] = graphNo
        if 'realizationNo' in kwargs.keys():
            varsFilename += 'R%02d' % realizationNo
            varsPickle['realizationNo'] = realizationNo
        # Create the file
        pathToFile = os.path.join(saveDir, varsFilename + '.pkl')
        # Open and save it
        with open(pathToFile, 'wb') as trainVarsFile:
            pickle.dump(varsPickle, trainVarsFile)

    # Now, if we didn't do any training (i.e. nEpochs = 0), then the last is
    # also the best.
    if nEpochs == 0:
        for key in modelsDict.keys():
            modelsDict[key].save(label = 'Best')
            modelsDict[key].save(label = 'Last')
        if doPrint:
            print("WARNING: No training. Best and Last models are the same.")

    # After training is done, reload best model before proceeding to evaluation:
    for key in modelsDict.keys():
        modelsDict[key].load(label = 'Best')

    #\\\ Print out best:
    if doPrint and nEpochs > 0:
        for key in modelsDict.keys():
            print("=> Best validation achieved for %s (E: %2d, B: %2d): %.4f" %(
                   key, bestEpoch[key] + 1, bestBatch[key] + 1, bestScore[key]))
