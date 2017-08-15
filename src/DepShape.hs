{-# LANGUAGE GADTs                #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE UndecidableInstances #-}

module DepShape where

import           Data.ByteString (ByteString)
import           Data.Constraint (Constraint)
import           Data.Int (Int64, Int8, Int16)
import           Data.Maybe (fromJust)
import           Data.Proxy (Proxy(..))
import qualified Data.Vector as VN
import           Data.Vector.Sized (Vector(..), toList, fromList)
import           Data.Word (Word8)
import           GHC.TypeLits (Nat, KnownNat, natVal)
import           GHC.TypeLits

import           TensorFlow.Core
import           TensorFlow.Core (Shape(..), TensorType, Tensor, Build)
import           TensorFlow.Ops (constant, add, matMul)
import           TensorFlow.Session (runSession, run)

instance Eq Shape where
  (==) (Shape s) (Shape r) = s == r

data MyShape (s :: [Nat]) where
  NilShape :: MyShape '[]
  (:--) :: KnownNat m => Proxy m -> MyShape s -> MyShape (m ': s)

data MyTensor v a (s :: [Nat]) where
  MyTensor :: (TensorType a) => Tensor v a -> MyTensor v a s

infixr 5 :--

class MkMyShape (s :: [Nat]) where
  mkMyShape :: MyShape s
instance MkMyShape '[] where
  mkMyShape = NilShape
instance (MkMyShape s, KnownNat m) => MkMyShape (m ': s) where
  mkMyShape = Proxy :-- mkMyShape

toShape :: MyShape s -> Shape
toShape NilShape = Shape []
toShape ((pm :: Proxy m) :-- s) = Shape (fromInteger (natVal pm) : s')
  where
    (Shape s') = toShape s

fromShape :: forall s. MkMyShape s => Shape -> Maybe (MyShape s)
fromShape shape = if toShape myShape == shape
  then Just myShape
  else Nothing
  where
    myShape = mkMyShape :: MyShape s

dConstant :: (TensorType a, ShapeProduct s ~ n) => Vector n a -> MyShape s -> MyTensor Build a s
dConstant elems shp = MyTensor $ constant (toShape shp) (toList elems)

dAdd :: (TensorType a, a /= Bool) => MyTensor Build a s -> MyTensor Build a s -> MyTensor Build a s
dAdd (MyTensor t1) (MyTensor t2) = MyTensor (t1 `add` t2)

dMatMul :: (TensorType a, a /= Bool, a /= Int8, a /= Int16, a /= Int64, a /= Word8, a /= ByteString)
   => MyTensor Build a '[i,n] -> MyTensor Build a '[n,o] -> MyTensor Build a '[i,o]
dMatMul (MyTensor t1) (MyTensor t2) = MyTensor (t1 `matMul` t2)

type family ShapeProduct (s :: [Nat]) :: Nat
type instance ShapeProduct '[] = 1
type instance ShapeProduct (m ': s) = m * ShapeProduct s

main :: IO (VN.Vector Int64)
main = runSession $ do
  let (shape1 :: MyShape '[2,2]) = fromJust $ fromShape (Shape [2,2])
  let (elems1 :: Vector 4 Int64) = fromJust $ fromList [1,2,3,4]
  let (elems2 :: Vector 4 Int64) = fromJust $ fromList [5,6,7,8]
  let (constant1 :: MyTensor Build Int64 '[2,2]) = dConstant elems1 shape1
  let (constant2 :: MyTensor Build Int64 '[2,2]) = dConstant elems2 shape1
  let (MyTensor additionNode) = constant1 `dAdd` constant2 
  run additionNode

main2 :: IO (VN.Vector Float)
main2 = runSession $ do
  let (shape1 :: MyShape '[4,3]) = fromJust $ fromShape (Shape [4,3])
  let (shape2 :: MyShape '[3,2]) = fromJust $ fromShape (Shape [3,2])
  let (elems1 :: Vector 12 Float) = fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
  let (elems2 :: Vector 6 Float) = fromJust $ fromList [5,6,7,8,9,10]
  let (constant1 :: MyTensor Build Float '[4,3]) = dConstant elems1 shape1
  let (constant2 :: MyTensor Build Float '[3,2]) = dConstant elems2 shape2
  let (MyTensor multNode) = constant1 `dMatMul` constant2
  run multNode
