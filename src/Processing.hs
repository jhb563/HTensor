{-# LANGUAGE DeriveGeneric   #-}
{-# LANGUAGE OverloadedLists #-}

module Processing where

import Data.ByteString.Lazy.Char8 (pack)
import Data.Csv (FromRecord, decode, HasHeader(..))
import Data.Int
import Data.Vector (Vector, length, fromList, (!), toList)
import Data.Word (Word8)
import GHC.Generics (Generic)
import System.Random.Shuffle (shuffleM)

import TensorFlow.Core (TensorData, encodeTensorData)

import Constants (irisFeatures, sampleSize, mnistFeatures)

data IrisRecord = IrisRecord
  { field1 :: Float
  , field2 :: Float
  , field3 :: Float
  , field4 :: Float
  , label  :: Int64
  }
  deriving (Show, Generic)

instance FromRecord IrisRecord

readIrisFromFile :: FilePath -> IO (Vector IrisRecord)
readIrisFromFile fp = do
  contents <- readFile fp
  let contentsAsBs = pack contents
  let results = decode HasHeader contentsAsBs :: Either String (Vector IrisRecord)
  case results of
    Left err -> error err
    Right records -> return records

-- A function that takes this vector of records, and selects 10 of them at random.
chooseRandomRecords :: Vector a -> IO (Vector a)
chooseRandomRecords records = do
  let numRecords = Data.Vector.length records
  chosenIndices <- take sampleSize <$> shuffleM [0..(numRecords - 1)]
  return $ fromList $ map (records !) chosenIndices

convertRecordsToTensorData :: Vector IrisRecord -> (TensorData Float, TensorData Int64)
convertRecordsToTensorData records = (input, output)
  where
    numRecords = Data.Vector.length records 
    input = encodeTensorData [fromIntegral numRecords, irisFeatures]
      (fromList $ concatMap recordToInputs records)
    output = encodeTensorData [fromIntegral numRecords] (label <$> records)
    recordToInputs :: IrisRecord -> [Float]
    recordToInputs rec = [field1 rec, field2 rec, field3 rec, field4 rec]

convertDigitRecordsToTensorData
  :: Vector (Vector Word8, Word8)
  -> (TensorData Float, TensorData Word8)
convertDigitRecordsToTensorData records = (input, output)
  where
    numRecords = Data.Vector.length records
    input = encodeTensorData [fromIntegral numRecords, mnistFeatures]
      (fromList $ concatMap recordToInputs records)
    output = encodeTensorData [fromIntegral numRecords] (snd <$> records)
    recordToInputs :: (Vector Word8, Word8) -> [Float]
    recordToInputs rec = fromIntegral <$> (toList . fst) rec
