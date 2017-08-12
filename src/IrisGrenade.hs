{-# LANGUAGE DataKinds #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE GADTs #-}

module IrisGrenade where

import           Control.Monad (foldM_)
import           Control.Monad.Random (MonadRandom)
import           Control.Monad.IO.Class (liftIO)
import           Data.Foldable (foldl')
import           Data.Maybe (mapMaybe)
import qualified Data.Vector.Storable as VS
import qualified Data.Vector as V
import           GHC.Float (float2Double)
import           Grenade
import           Grenade.Core.LearningParameters (LearningParameters(..))
import           Grenade.Core.Shape (fromStorable)
import           Grenade.Utils.OneHot (oneHot)
import           Numeric.LinearAlgebra (maxIndex)
import           Numeric.LinearAlgebra.Static (extract)

import           Processing (IrisRecord(..), readIrisFromFile, chooseRandomRecords)

type IrisNetwork = Network 
  '[FullyConnected 4 10, Relu, FullyConnected 10 3]
  '[ 'D1 4, 'D1 10, 'D1 10, 'D1 3]

type IrisRow = (S ('D1 4), S ('D1 3))

randomIris :: MonadRandom m => m IrisNetwork
randomIris = randomNetwork

runIris :: FilePath -> FilePath -> IO ()
runIris trainingFile testingFile = do
  trainingRecords <- liftIO $ readIrisFromFile trainingFile
  testRecords <- liftIO $ readIrisFromFile testingFile
  
  let trainingData = mapMaybe parseRecord (V.toList trainingRecords)
  let testData = mapMaybe parseRecord (V.toList testRecords)
  
  if length trainingData /= length trainingRecords || length testData /= length testRecords
    then putStrLn "Hmmm there were some problems parsing the data"
    else do
      initialNetwork <- randomIris
      foldM_ (run trainingData testData) initialNetwork [1..100]
  
  where
    learningParams :: LearningParameters
    learningParams = LearningParameters 0.01 0.9 0.0005

    trainRow :: LearningParameters -> IrisNetwork -> IrisRow -> IrisNetwork
    trainRow lp network (input, output) = train lp network input output

    run :: [IrisRow] -> [IrisRow] -> IrisNetwork -> Int -> IO IrisNetwork
    run trainData testData network iterationNum = do
      -- Slower drop the learning rate
      sampledRecords <- V.toList <$> chooseRandomRecords (V.fromList trainData)
      let revisedParams = learningParams 
            { learningRate = learningRate learningParams * 0.99 ^ iterationNum}
      let newNetwork = foldl' (trainRow revisedParams) network sampledRecords
      let labelVectors = fmap (testRow newNetwork) testData
      let labelValues = fmap getLabels labelVectors
      let total = length labelValues
      let correctEntries = length $ filter ((==) <$> fst <*> snd) labelValues
      putStrLn $ "Iteration: " ++ show iterationNum
      putStrLn $ show correctEntries ++ " correct out of: " ++ show total
      return newNetwork

    -- Takes a test row, returns predicted output and actual output from the network.
    testRow :: IrisNetwork -> IrisRow -> (S ('D1 3), S ('D1 3))
    testRow net (rowInput, predictedOutput) = (predictedOutput, runNet net rowInput)

    -- Goes from probability output vector to label
    getLabels :: (S ('D1 3), S ('D1 3)) -> (Int, Int)
    getLabels (S1D predictedLabel, S1D actualOutput) = 
      (maxIndex (extract predictedLabel), maxIndex (extract actualOutput))

parseRecord :: IrisRecord -> Maybe IrisRow
parseRecord record = case (input, output) of
  (Just i, Just o) -> Just (i, o)
  _ -> Nothing
  where
    input = fromStorable $ VS.fromList $ float2Double <$>
      [ field1 record / 8.0, field2 record / 8.0, field3 record / 8.0, field4 record / 8.0]
    output = oneHot (fromIntegral $ label record)
