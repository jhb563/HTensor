module Main where

import Digits (runDigits)

main :: IO ()
main = runDigits 
  "data/mnist-train-images.gz"
  "data/mnist-train-labels.gz"
  "data/mnist-test-images.gz"
  "data/mnist-test-labels.gz"
