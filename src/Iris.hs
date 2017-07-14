{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings   #-}

module Iris 
  ( runIris)
  where

import Data.ByteString (ByteString)
import Control.Monad (forM_, when)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64)
import Data.Text

import TensorFlow.Core (TensorData, Session, Build, render, runWithFeeds, feed, unScalar, build,
                        Tensor, withNameScope)
import TensorFlow.Logging (withEventWriter, logGraph, logSummary, scalarSummary,
                           mergeAllSummaries, SummaryTensor, histogramSummary)
import TensorFlow.Minimize (minimizeWith, adam)
import TensorFlow.Ops (placeholder, truncatedNormal, add, matMul, relu,
                      argMax, scalar, cast, oneHot, reduceMean, softmaxCrossEntropyWithLogits, 
                      equal, vector)
import TensorFlow.Session (runSession)
import TensorFlow.Variable (readValue, initializedVariable, Variable)
import Data.ProtoLens.Encoding (decodeMessageOrDie)

import Constants (irisFeatures, irisLabels)
import Processing (readIrisFromFile, convertRecordsToTensorData, chooseRandomRecords)

data Model = Model
  { train :: TensorData Float
          -> TensorData Int64
          -> Session ()
  , errorRate :: TensorData Float
              -> TensorData Int64
              -> SummaryTensor
              -> Session (Float, ByteString)
  }

{-randomParam :: Int64 -> Shape -> Build (Tensor Build Float)
randomParam width (Shape shape) =
    (`mul` stddev) <$> truncatedNormal (vector shape)
  where
    stddev = scalar (1 / sqrt (fromIntegral width))-}

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
  let numHiddenUnits = 10

  inputs <- placeholder [batchSize, irisFeatures]
  outputs <- placeholder [batchSize]

  (hiddenWeights, hiddenBiases, hiddenResults) <- 
    buildNNLayer irisFeatures numHiddenUnits inputs "layer1"
  let rectifiedHiddenResults = relu hiddenResults
  (finalWeights, finalBiases, finalResults) <-
    buildNNLayer numHiddenUnits irisLabels rectifiedHiddenResults "layer2"
  histogramSummary "Weights" (readValue finalWeights)

  (errorRate_, train_) <- withNameScope "error_calculation" $ do
    actualOutput <- render $ cast $ argMax finalResults (scalar (1 :: Int64))
    let correctPredictions = equal actualOutput outputs
    er <- render $ 1 - (reduceMean (cast correctPredictions))
    scalarSummary "Error" er

    let outputVectors = oneHot outputs (fromIntegral irisLabels) 1 0
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
eventsDir = "/tmp/tensorflow/iris/logs/"

runIris :: FilePath -> FilePath -> IO ()
runIris trainingFile testingFile = withEventWriter eventsDir $ \eventWriter -> runSession $ do
  trainingRecords <- liftIO $ readIrisFromFile trainingFile
  testRecords <- liftIO $ readIrisFromFile testingFile

  model <- build createModel
  logGraph eventWriter createModel
  summaryTensor <- build mergeAllSummaries
  
  -- Training
  forM_ ([0..1000] :: [Int]) $ \i -> do
    trainingSample <- liftIO $ chooseRandomRecords trainingRecords
    let (trainingInputs, trainingOutputs) = convertRecordsToTensorData trainingSample
    (train model) trainingInputs trainingOutputs
    when (i `mod` 100 == 0) $ do
      (err, summaryBytes) <- (errorRate model) trainingInputs trainingOutputs summaryTensor
      let summary = decodeMessageOrDie summaryBytes
      liftIO $ putStrLn $ "Current training error " ++ show (err * 100)
      logSummary eventWriter (fromIntegral i) summary

  liftIO $ putStrLn ""

  -- Testing
  let (testingInputs, testingOutputs) = convertRecordsToTensorData testRecords
  (testingError, _) <- (errorRate model) testingInputs testingOutputs summaryTensor
  liftIO $ putStrLn $ "test error " ++ show (testingError * 100)

  return ()
