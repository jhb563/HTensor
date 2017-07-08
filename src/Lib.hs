{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
    ( runSimple
    , runPlaceholder
    , runVariable
    ) where

import TensorFlow.Core (Tensor, Value, feed, encodeTensorData, Scalar(..))
import TensorFlow.Ops (constant, add, placeholder, sub, reduceSum, mul)
import TensorFlow.GenOps.Core (square)
import TensorFlow.Variable (readValue, initializedVariable, Variable)
import TensorFlow.Session (runSession, run, runWithFeeds)
import TensorFlow.Minimize (gradientDescent, minimizeWith)

import Control.Monad (replicateM_)
import qualified Data.Vector as Vector
import Data.Vector (Vector)

runSimple :: IO (Vector Float)
runSimple = runSession $ do
  let node1 = constant [1] [3 :: Float]
  let node2 = constant [1] [4 :: Float]
  let additionNode = node1 `add` node2
  run additionNode

runPlaceholder :: Vector Float -> Vector Float -> IO (Vector Float)
runPlaceholder input1 input2 = runSession $ do
  (node1 :: Tensor Value Float) <- placeholder [1]
  (node2 :: Tensor Value Float) <- placeholder [1]
  let adderNode = node1 `add` node2
  let runStep = \node1Feed node2Feed -> runWithFeeds 
        [ feed node1 node1Feed
        , feed node2 node2Feed
        ] 
        adderNode
  runStep (encodeTensorData [1] input1) (encodeTensorData [1] input2)

runVariable :: Vector Float -> Vector Float -> IO (Float, Float)
runVariable xInput yInput = runSession $ do
  let xSize = fromIntegral $ Vector.length xInput
  let ySize = fromIntegral $ Vector.length yInput
  (w :: Variable Float) <- initializedVariable 3
  (b :: Variable Float) <- initializedVariable 1
  (x :: Tensor Value Float) <- placeholder [xSize]
  let linear_model = ((readValue w) `mul` x) `add` (readValue b)
  (y :: Tensor Value Float) <- placeholder [ySize]
  let square_deltas = square (linear_model `sub` y)
  let loss = reduceSum square_deltas
  trainStep <- minimizeWith (gradientDescent 0.01) loss [w,b] 
  let trainWithFeeds = \xF yF -> runWithFeeds
        [ feed x xF
        , feed y yF
        ]
        trainStep
  replicateM_ 1000 
    (trainWithFeeds (encodeTensorData [xSize] xInput) (encodeTensorData [ySize] yInput))
  (Scalar w_learned, Scalar b_learned) <- run (readValue w, readValue b)
  return (w_learned, b_learned)
