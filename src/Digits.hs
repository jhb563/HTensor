{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE OverloadedStrings   #-}

module Digits where

import Control.Monad (forM_, when)
import Control.Monad.IO.Class (liftIO)
import Data.ByteString (ByteString)
import Data.Int (Int64)
import Data.Text (Text)
import Data.Vector (fromList)
import Data.Word (Word8)

import TensorFlow.Core (Build, render, runWithFeeds, feed, unScalar, build,
                        Tensor, withNameScope, TensorData)
import TensorFlow.Logging (withEventWriter, logGraph, logSummary, scalarSummary,
                           mergeAllSummaries, histogramSummary, SummaryTensor)
import TensorFlow.Minimize (minimizeWith, adam)
import TensorFlow.Ops (placeholder, truncatedNormal, add, matMul, relu,
                      argMax, scalar, cast, oneHot, reduceMean, softmaxCrossEntropyWithLogits, 
                      equal, vector)
import TensorFlow.Session (runSession, Session)
import TensorFlow.Variable (readValue, initializedVariable, Variable)
import Data.ProtoLens.Encoding (decodeMessageOrDie)
import TensorFlow.Examples.MNIST.Parse (readMNISTSamples, readMNISTLabels)

import Constants (mnistFeatures, mnistLabels)
import Processing (convertDigitRecordsToTensorData, chooseRandomRecords)

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

createModel :: Build Model
createModel = do
  let batchSize = -1 -- Allows variable sized batches
  let numHiddenUnits = 1024

  inputs <- placeholder [batchSize, mnistFeatures]
  outputs <- placeholder [batchSize]

  (hiddenWeights, hiddenBiases, hiddenResults) <- 
    buildNNLayer mnistFeatures numHiddenUnits inputs "layer1"
  let rectifiedHiddenResults = relu hiddenResults
  (finalWeights, finalBiases, finalResults) <-
    buildNNLayer numHiddenUnits mnistLabels rectifiedHiddenResults "layer2"
  histogramSummary "Weights" (readValue finalWeights)

  (errorRate_, train_) <- withNameScope "error_calculation" $ do
    actualOutput <- render $ cast $ argMax finalResults (scalar (1 :: Int64))
    let correctPredictions = equal actualOutput outputs
    er <- render $ 1 - (reduceMean (cast correctPredictions))
    scalarSummary "Error" er

    let outputVectors = oneHot outputs (fromIntegral mnistLabels) 1 0
    let loss = reduceMean $ fst $ softmaxCrossEntropyWithLogits finalResults outputVectors
    let params = [hiddenWeights, hiddenBiases, finalWeights, finalBiases]
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
    forM_ ([0..1000] :: [Int]) $ \i -> do
      trainingSample <- liftIO $ chooseRandomRecords trainingRecords
      let (trainingInputs, trainingOutputs) = convertDigitRecordsToTensorData trainingSample
      (train model) trainingInputs trainingOutputs
      when (i `mod` 100 == 0) $ do
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
