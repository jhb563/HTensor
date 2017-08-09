{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}

module SingPractice where

-- import Data.Singletons

{-data OperatingSystem where
  MacOS   :: OperatingSystem 'True
  Linux   :: OperatingSystem 'True
  Windows :: OperatingSystem 'False-}

{-class IsUnixLike a where
  isUnixLike :: a -> Bool 

instance IsUnixLike OperatingSystem where
  isUnixLike MacOS = True
  isUnixLike Linux = True
  isUnixLike Windows = False

unixMsg :: (IsUnixLike a) => a -> String
unixMsg val = if isUnixLike val
  then "Like Unix!"
  else "Not like Unix!"

-- Restrict this to only unix!
writeOutput :: OperatingSystem -> String
writeOutput Windows = error "Panic!"
writeOutput _ = "Like unix!"-}
