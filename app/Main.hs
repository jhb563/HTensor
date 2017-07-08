{-# LANGUAGE OverloadedLists #-}

module Main where

import Lib

main :: IO ()
main = do
  result <- runSimple
  print result
  result1 <- runPlaceholder [3.0] [4.5]
  result2 <- runPlaceholder [2.7] [8.9]
  print result1
  print result2
  results <- runVariable [1.0, 2.0, 3.0, 4.0] [4.0, 9.0, 14.0, 19.0]
  print results
