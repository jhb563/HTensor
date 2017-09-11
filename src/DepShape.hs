{-# LANGUAGE GADTs                #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}

module DepShape where

import           Data.ByteString (ByteString)
import           Data.Int (Int64, Int8, Int16)
import           Data.Maybe (fromJust)
import           Data.Proxy (Proxy(..))
import qualified Data.Vector as VN
import           Data.Vector.Sized (Vector, toList, fromList)
import           Data.Word (Word8)
import           GHC.TypeLits (Nat, KnownNat, natVal)
import           GHC.TypeLits

import           TensorFlow.Core
import           TensorFlow.Core (Shape(..), TensorType, Tensor, Build)
import           TensorFlow.Ops (constant, add, matMul)
import           TensorFlow.Session (runSession, run)

instance Eq Shape where
  (==) (Shape s) (Shape r) = s == r

data SafeShape (s :: [Nat]) where
  NilShape :: SafeShape '[]
  (:--) :: KnownNat m => Proxy m -> SafeShape s -> SafeShape (m ': s)

data SafeTensor v a (s :: [Nat]) where
  SafeTensor :: (TensorType a) => Tensor v a -> SafeTensor v a s

infixr 5 :--

class MkSafeShape (s :: [Nat]) where
  mkSafeShape :: SafeShape s
instance MkSafeShape '[] where
  mkSafeShape = NilShape
instance (MkSafeShape s, KnownNat m) => MkSafeShape (m ': s) where
  mkSafeShape = Proxy :-- mkSafeShape

toShape :: SafeShape s -> Shape
toShape NilShape = Shape []
toShape ((pm :: Proxy m) :-- s) = Shape (fromInteger (natVal pm) : s')
  where
    (Shape s') = toShape s

fromShape :: forall s. MkSafeShape s => Shape -> Maybe (SafeShape s)
fromShape shape = if toShape myShape == shape
  then Just myShape
  else Nothing
  where
    myShape = mkSafeShape :: SafeShape s

safeConstant :: (TensorType a, ShapeProduct s ~ n) => Vector n a -> SafeShape s -> SafeTensor Build a s
safeConstant elems shp = SafeTensor $ constant (toShape shp) (toList elems)

safeAdd :: (TensorType a, a /= Bool) => SafeTensor Build a s -> SafeTensor Build a s -> SafeTensor Build a s
safeAdd (SafeTensor t1) (SafeTensor t2) = SafeTensor (t1 `add` t2)

safeMatMul :: (TensorType a, a /= Bool, a /= Int8, a /= Int16, a /= Int64, a /= Word8, a /= ByteString)
   => SafeTensor Build a '[i,n] -> SafeTensor Build a '[n,o] -> SafeTensor Build a '[i,o]
safeMatMul (SafeTensor t1) (SafeTensor t2) = SafeTensor (t1 `matMul` t2)

type family ShapeProduct (s :: [Nat]) :: Nat
type instance ShapeProduct '[] = 1
type instance ShapeProduct (m ': s) = m * ShapeProduct s

main :: IO (VN.Vector Int64)
main = runSession $ do
  let (shape1 :: SafeShape '[2,2]) = fromJust $ fromShape (Shape [2,2])
  let (elems1 :: Vector 4 Int64) = fromJust $ fromList [1,2,3,4]
  let (constant1 :: SafeTensor Build Int64 '[2,2]) = safeConstant elems1 shape1
  let (SafeTensor t) = constant1
  run t

main2 :: IO (VN.Vector Float)
main2 = runSession $ do
  let (shape1 :: SafeShape '[4,3]) = fromJust $ fromShape (Shape [4,3])
  let (shape2 :: SafeShape '[3,2]) = fromJust $ fromShape (Shape [3,2])
  let (shape3 :: SafeShape '[4,2]) = fromJust $ fromShape (Shape [4,2])
  let (elems1 :: Vector 12 Float) = fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
  let (elems2 :: Vector 6 Float) = fromJust $ fromList [5,6,7,8,9,10]
  let (elems3 :: Vector 8 Float) = fromJust $ fromList [11,12,13,14,15,16,17,18]
  let (constant1 :: SafeTensor Build Float '[4,3]) = safeConstant elems1 shape1
  let (constant2 :: SafeTensor Build Float '[3,2]) = safeConstant elems2 shape2
  let (constant3 :: SafeTensor Build Float '[4,2]) = safeConstant elems3 shape3
  let (multTensor :: SafeTensor Build Float '[4,2]) = constant1 `safeMatMul` constant2
  let (addTensor :: SafeTensor Build Float '[4,2]) = multTensor `safeAdd` constant3
  let (SafeTensor finalTensor) = addTensor
  run finalTensor
