module Main where

import Iris (runIris)

main :: IO ()
main = runIris "data/iris_training.csv" "data/iris_test.csv"
