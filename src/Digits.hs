{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}

module Digits where

import Control.Lens ((.~))
import Control.Monad (forM_, when)
import Control.Monad.IO.Class (liftIO)
import Data.ByteString (ByteString)
import Data.Int (Int64, Int32)
import Data.Text (Text)
import Data.Vector (fromList)
import Data.Word (Word8)

import TensorFlow.Core (Build, render, runWithFeeds, feed, unScalar, build,
                        Tensor, withNameScope, TensorData, opAttr)
import TensorFlow.Logging (withEventWriter, logGraph, logSummary, scalarSummary,
                           mergeAllSummaries, SummaryTensor)
import TensorFlow.Minimize (minimizeWith, adam)
import TensorFlow.Ops (placeholder, truncatedNormal, add, matMul, relu,
                      argMax, scalar, cast, oneHot, reduceMean, softmaxCrossEntropyWithLogits, 
                      equal, vector, reshape)
import TensorFlow.Session (runSession, Session)
import TensorFlow.Variable (readValue, initializedVariable, Variable)
import Data.ProtoLens.Encoding (decodeMessageOrDie)
import TensorFlow.Examples.MNIST.Parse (readMNISTSamples, readMNISTLabels)
import TensorFlow.GenOps.Core (conv2D', maxPool')

import Constants (mnistFeatures, mnistLabels)
import Processing (convertDigitRecordsToTensorData, chooseRandomRecords)

patchSize :: Int64
patchSize = 5

imageDimen :: Int32
imageDimen = 28

data Model = Model
  { train :: TensorData Float
          -> TensorData Word8
          -> Session ()
  , errorRate :: TensorData Float
              -> TensorData Word8
              -> SummaryTensor
              -> Session (Float, ByteString)
  }

buildNNLayer :: Int64 -> Int64 -> Tensor v Float -> Text
             -> Build (Variable Float, Variable Float, Tensor Build Float)
buildNNLayer inputSize outputSize input layerName = withNameScope layerName $ do
  weights <- truncatedNormal (vector [inputSize, outputSize]) >>= initializedVariable
  bias <- truncatedNormal (vector [outputSize]) >>= initializedVariable
  let results = (input `matMul` readValue weights) `add` readValue bias
  return (weights, bias, results)

buildConvPoolLayer :: Int64 -> Int64 -> Tensor v Float -> Text
                   -> Build (Variable Float, Variable Float, Tensor Build Float)
buildConvPoolLayer inputChannels outputChannels input layerName = withNameScope layerName $ do
  weights <- truncatedNormal (vector weightsShape)
    >>= initializedVariable
  bias <- truncatedNormal (vector [outputChannels]) >>= initializedVariable
  let conv = conv2D' convAttrs input (readValue weights)
        `add` readValue bias
  let results = maxPool' poolAttrs (relu conv)
  return (weights, bias, results)
  where
    weightsShape :: [Int64]
    weightsShape = [patchSize, patchSize, inputChannels, outputChannels]
    convStridesAttr = opAttr "strides" .~ ([1,1,1,1] :: [Int64])
    poolStridesAttr = opAttr "strides" .~ ([1,2,2,1] :: [Int64])
    poolKSizeAttr = opAttr "ksize" .~ ([1,2,2,1] :: [Int64])
    paddingAttr = opAttr "padding" .~ ("SAME" :: ByteString)
    dataFormatAttr = opAttr "data_format" .~ ("NHWC" :: ByteString)
    convAttrs = convStridesAttr . paddingAttr . dataFormatAttr
    poolAttrs = poolKSizeAttr . poolStridesAttr . paddingAttr . dataFormatAttr

createModel :: Build Model
createModel = do
  let batchSize = -1 -- Allows variable sized batches
  let conv1OutputChannels = 32
  let conv2OutputChannels = 64
  let denseInputSize = 7 * 7 * 64 :: Int32
  let numHiddenUnits = 1024

  inputs <- placeholder [batchSize, mnistFeatures]
  outputs <- placeholder [batchSize]

  let inputImage = reshape inputs (vector [-1, imageDimen, imageDimen, 1])

  (convWeights1, convBiases1, convResults1) <- 
    buildConvPoolLayer 1 conv1OutputChannels inputImage "convLayer1"
  (convWeights2, convBiases2, convResults2) <-
    buildConvPoolLayer conv1OutputChannels conv2OutputChannels convResults1 "convLayer2"

  let denseInput = reshape convResults2 (vector [-1, denseInputSize])
  (denseWeights1, denseBiases1, denseResults1) <-
    buildNNLayer (fromIntegral denseInputSize) numHiddenUnits denseInput "denseLayer1"  
  let rectifiedDenseResults = relu denseResults1
  (denseWeights2, denseBiases2, denseResults2) <-
    buildNNLayer numHiddenUnits mnistLabels rectifiedDenseResults "denseLayer2"

  (errorRate_, train_) <- withNameScope "error_calculation" $ do
    actualOutput <- render $ cast $ argMax denseResults2 (scalar (1 :: Int64))
    let correctPredictions = equal actualOutput outputs
    er <- render $ 1 - (reduceMean (cast correctPredictions))
    scalarSummary "Error" er

    let outputVectors = oneHot outputs (fromIntegral mnistLabels) 1 0
    let loss = reduceMean $ fst $ softmaxCrossEntropyWithLogits denseResults2 outputVectors
    let params = [ convWeights1, convBiases1, convWeights2, convBiases2,
                   denseWeights1, denseBiases1, denseWeights2, denseBiases2 ]
    tr <- minimizeWith adam loss params
    return (er, tr)

  return $ Model
    { train = \inputFeed outputFeed -> 
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          train_
    , errorRate = \inputFeed outputFeed summaryTensor -> do
        (errorTensorResult, summaryTensorResult) <- runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          (errorRate_, summaryTensor)
        return (unScalar errorTensorResult, unScalar summaryTensorResult)
    }

eventsDir :: FilePath
eventsDir = "/tmp/tensorflow/mnist/logs/"

runDigits :: FilePath -> FilePath -> FilePath -> FilePath -> IO ()
runDigits trainImageFile trainLabelFile testImageFile testLabelFile = 
  withEventWriter eventsDir $ \eventWriter -> runSession $ do
    -- trainingImages, testImages :: [Vector Word8]
    trainingImages <- liftIO $ readMNISTSamples trainImageFile
    testImages <- liftIO $ readMNISTSamples testImageFile
    -- traininglabels, testLabels :: [Word8]
    trainingLabels <- liftIO $ readMNISTLabels trainLabelFile
    testLabels <- liftIO $ readMNISTLabels testLabelFile
    -- trainingRecords, testRecords :: Vector (Vector Word8, Word8)
    let trainingRecords = fromList $ zip trainingImages trainingLabels
    let testRecords = fromList $ zip testImages testLabels

    model <- build createModel
    logGraph eventWriter createModel
    summaryTensor <- build mergeAllSummaries
    
    -- Training
    forM_ ([0..20000] :: [Int]) $ \i -> do
      trainingSample <- liftIO $ chooseRandomRecords trainingRecords
      let (trainingInputs, trainingOutputs) = convertDigitRecordsToTensorData trainingSample
      (train model) trainingInputs trainingOutputs
      when (i `mod` 1000 == 0) $ do
        (err, summaryBytes) <- (errorRate model) trainingInputs trainingOutputs summaryTensor
        let summary = decodeMessageOrDie summaryBytes
        liftIO $ putStrLn $ "Current training error " ++ show (err * 100)
        logSummary eventWriter (fromIntegral i) summary

    liftIO $ putStrLn ""

    -- Testing
    let (testingInputs, testingOutputs) = convertDigitRecordsToTensorData testRecords
    (testingError, _) <- (errorRate model) testingInputs testingOutputs summaryTensor
    liftIO $ putStrLn $ "test error " ++ show (testingError * 100)

    return ()
