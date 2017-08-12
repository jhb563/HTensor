module Main where

import IrisGrenade (runIris)

main :: IO ()
main = runIris "data/iris_training.csv" "data/iris_test.csv"
