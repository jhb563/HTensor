{-# LANGUAGE OverloadedLists #-}

module Iris 
  ( runIris)
  where

import Control.Monad (forM_, when)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64)

import TensorFlow.Core (TensorData, Session, Build, render, runWithFeeds, feed, unScalar, build,
                        Tensor)
import TensorFlow.Minimize (minimizeWith, adam)
import TensorFlow.Ops (placeholder, truncatedNormal, add, matMul, relu,
                      argMax, scalar, cast, oneHot, reduceMean, softmaxCrossEntropyWithLogits, 
                      equal, vector)
import TensorFlow.Session (runSession)
import TensorFlow.Variable (readValue, initializedVariable, Variable)

import Constants (irisFeatures, irisLabels)
import Processing (readIrisFromFile, convertRecordsToTensorData, chooseRandomRecords)

data Model = Model
  { train :: TensorData Float
          -> TensorData Int64
          -> Session ()
  , errorRate :: TensorData Float
              -> TensorData Int64
              -> Session Float
  }

{-randomParam :: Int64 -> Shape -> Build (Tensor Build Float)
randomParam width (Shape shape) =
    (`mul` stddev) <$> truncatedNormal (vector shape)
  where
    stddev = scalar (1 / sqrt (fromIntegral width))-}

buildNNLayer :: Int64 -> Int64 -> Tensor v Float
             -> Build (Variable Float, Variable Float, Tensor Build Float)
buildNNLayer inputSize outputSize input = do
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
    buildNNLayer irisFeatures numHiddenUnits inputs
  let rectifiedHiddenResults = relu hiddenResults
  (finalWeights, finalBiases, finalResults) <-
    buildNNLayer numHiddenUnits irisLabels rectifiedHiddenResults

  actualOutput <- render $ cast $ argMax finalResults (scalar (1 :: Int64))
  let correctPredictions = equal actualOutput outputs
  errorRate_ <- render $ 1 - (reduceMean (cast correctPredictions))

  let outputVectors = oneHot outputs (fromIntegral irisLabels) 1 0
  let loss = reduceMean $ fst $ softmaxCrossEntropyWithLogits finalResults outputVectors
  let params = [hiddenWeights, hiddenBiases, finalWeights, finalBiases]
  train_ <- minimizeWith adam loss params

  return $ Model
    { train = \inputFeed outputFeed -> 
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          train_
    , errorRate = \inputFeed outputFeed -> unScalar <$>
        runWithFeeds
          [ feed inputs inputFeed
          , feed outputs outputFeed
          ]
          errorRate_
    }

runIris :: FilePath -> FilePath -> IO ()
runIris trainingFile testingFile = runSession $ do
  trainingRecords <- liftIO $ readIrisFromFile trainingFile
  testRecords <- liftIO $ readIrisFromFile testingFile

  model <- build createModel
  
  -- Training
  forM_ ([0..1000] :: [Int]) $ \i -> do
    trainingSample <- liftIO $ chooseRandomRecords trainingRecords
    let (trainingInputs, trainingOutputs) = convertRecordsToTensorData trainingSample
    (train model) trainingInputs trainingOutputs
    when (i `mod` 100 == 0) $ do
      err <- (errorRate model) trainingInputs trainingOutputs
      liftIO $ putStrLn $ "Current training error " ++ show (err * 100)

  liftIO $ putStrLn ""

  -- Testing
  let (testingInputs, testingOutputs) = convertRecordsToTensorData testRecords
  testingError <- (errorRate model) testingInputs testingOutputs
  liftIO $ putStrLn $ "test error " ++ show (testingError * 100)

  return ()
