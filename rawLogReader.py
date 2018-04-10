import os
import glob
import pandas as pd
import numpy as np

def readLogDirectoryWriteCompiled(inputDirectory, outputDirectory):
    # replace this with your appropriate folder
    allFiles = glob.glob(os.path.join(inputDirectory,"*.txt"))

    settingsInfoHeaderRow = ['logName', 'BatchSize', 'Dataset', 'bitsW',
                             'bitsA', 'bitsG', 'bitsE',  'bitsR,', 'lr',
                             'lr_sched', 'L2', 'lossFunc', 'optimizer',
                             'numEpochs', 'numLayers']

    settingsDf = pd.DataFrame(columns=settingsInfoHeaderRow)

    runEpochInfoHeader = ['logName', 'Loss', 'Train_Error', 'Test_Error']
    runEpochInfoDf = pd.DataFrame(columns=runEpochInfoHeader)

    dummyReadHeader = range(0,15)
    fileCount = 0

    for file in allFiles:
        layerCount = 0
        fileCount += 1
        settingsRow = []
        logDf = pd.read_csv(file, names=dummyReadHeader, delimiter=' ', engine='python')

        # finding name of file
        logTitleString = ''
        for index, row in logDf.iterrows():
            if '2018' in str(row[0]):
                for x in row:
                    if type(x) == str:
                        logTitleString += str(x) +'_'
                    else:
                        logTitleString = logTitleString[:-1]
                        break
                settingsRow.append(logTitleString)
            elif row[0] == 'batchSize':
                settingsRow.append(row[2])
            elif row[0] =='dataSet':
                settingsRow.append(row[2])
            elif row[0] == 'bitsW':
                settingsRow.append(row[2])
            elif row[0] == 'bitsA':
                settingsRow.append(row[2])
            elif row[0] == 'bitsG':
                settingsRow.append(row[2])
            elif row[0] == 'bitsE':
                settingsRow.append(row[2])
            elif row[0] == 'bitsR':
                settingsRow.append(row[2])
            elif row[0] == 'lr':
                lr_str = ''
                for x in row:
                    if x != None:
                         lr_str += str(x)
                    else:
                        break
                settingsRow.append(lr_str)
            elif row[0] == 'lr_schedule':
                lr_sched_str = ''
                for x in row:
                    if x != None:
                         lr_sched_str += str(x)
                    else:
                        break
                settingsRow.append(lr_sched_str)
            elif row[0] == 'L2':
                settingsRow.append(row[2])
            elif row[0] == 'lossFunc':
                settingsRow.append(row[2])
            elif row[0] == 'optimizer':
                lr_sched_str = ''
                for x in row:
                    if x != None or x!='#':
                         lr_sched_str += str(x)
                    else:
                        break
                settingsRow.append(lr_sched_str)
            elif 'numEpochs' in str(row[0]):
                settingsRow.append(row[2])
            #prep count for numLayers Header column
            elif row[0] == 'W:':
                layerCount += 1
            #ending condition, this is about run info
            elif row[0] == 'Epoch:':
                break
        settingsRow.append(int(layerCount)/2)
        settingsDf.loc[len(settingsDf)] = settingsRow

        #building Epoch Information everything else
        for index, row in logDf.iterrows():
            if row[0] == 'Epoch:':
                dfRow = [logTitleString]
                #append loss error
                dfRow.append(row[4])
                #append train error
                dfRow.append(row[6])
                #append Test error
                dfRow.append(row[8])

                runEpochInfoDf.loc[len(runEpochInfoDf)] = dfRow

            # settingsInfoHeaderRow = ['logName', 'BatchSize', 'Dataset', 'bitsW', 'bitsA', 'bitsG',
            #                          'bitsE', 'lr_sched', 'L2', 'lossFunc', 'optimizer', 'numEpochs']
            #checking title
            #elif '2018' in row[0]:
        print('Done File', logTitleString)
    settingsDfFile = outputDirectory + 'RawSettings.csv'
    settingsDf.to_csv(settingsDfFile, index=False)

    runEpochFile = outputDirectory + 'RawEpochInfo.csv'
    runEpochInfoDf.to_csv(runEpochFile, index=False)

def removeBrokenSessions(settingsDf, epochsDf):
    print('quoi')
    validEpochSessionsDf = None
    validSettingsDf = pd.DataFrame(columns=list(settingsDf.columns.values))
    brokenSettingsDf = pd.DataFrame(columns=list(settingsDf.columns.values))
    for index, row in settingsDf.iterrows():
        logDf = epochsDf.loc[epochsDf['logName'] == row['logName']]
        dfMode = logDf['Train_Error'].mode()
        if len(dfMode) >= 1 and dfMode[0] != 0:
            validSettingsDf.loc[len(validSettingsDf)] = row
            if validEpochSessionsDf is None:
                validEpochSessionsDf = logDf
            else:
                validEpochSessionsDf = pd.concat((validEpochSessionsDf, logDf), ignore_index=True)
        else:
            brokenSettingsDf.loc[len(brokenSettingsDf)] = row

    return brokenSettingsDf, validEpochSessionsDf, validSettingsDf

def plotTrainingEpochsSession(trainingKey, epochDf):
    print('fuck me')

if __name__ == "__main__":
    inputLogDirectory = 'C:\\Users\\jw\\Documents\\Classes\\AML\\logAnalysis\\rawLogData'
    compiledDirectory = 'compiledData\\'
    #readLogDirectoryWriteCompiled(inputLogDirectory, compiledDirectory)
    settingsFile = compiledDirectory + 'rawSettings.csv'
    epochFile = compiledDirectory + 'rawEpochInfo.csv'

    rawSettingsDf = pd.read_csv(settingsFile, index_col=None)
    rawEpochDf = pd.read_csv(epochFile, index_col=None)

    brokenSettingsDf, cleanedEpochsDf, cleanedSettingsDf = removeBrokenSessions(rawSettingsDf, rawEpochDf)

    brokenSettingsDf.to_csv((compiledDirectory + 'BrokenSettings.csv'), index=False)
    cleanedSettingsDf.to_csv((compiledDirectory + 'CleanedSettings.csv'), index=False)
    cleanedEpochsDf.to_csv((compiledDirectory + 'CleanedEpochs.csv'), index=False)


    print('Done All Files')


