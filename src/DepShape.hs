{-# LANGUAGE GADTs                #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE UndecidableInstances #-}

module DepShape where

import           Data.ByteString (ByteString)
import           Data.Constraint (Constraint)
import           Data.Int (Int64, Int8, Int16)
import           Data.Maybe (fromJust)
import           Data.Proxy (Proxy(..))
import           Data.Type.List (Union)
import qualified Data.Vector as VN
import           Data.Vector.Sized (Vector(..), toList, fromList)
import           Data.Word (Word8)
import           GHC.TypeLits (Nat, KnownNat, natVal)
import           GHC.TypeLits

import           TensorFlow.Core
import           TensorFlow.Core (Shape(..), TensorType, Tensor, Build)
import           TensorFlow.Ops (constant, add, matMul, placeholder)
import           TensorFlow.Session (runSession, run)

instance Eq Shape where
  (==) (Shape s) (Shape r) = s == r

data SafeShape (s :: [Nat]) where
  NilShape :: SafeShape '[]
  (:--) :: KnownNat m => Proxy m -> SafeShape s -> SafeShape (m ': s)

data SafeTensor v a (s :: [Nat]) (p :: [(Symbol, [Nat])]) where
  SafeTensor :: (TensorType a) => Tensor v a -> SafeTensor v a s p

data SafeTensorData a (l :: Symbol) (s :: [Nat]) where
  SafeTensorData :: (TensorType a) => TensorData a -> SafeTensorData a l s

infixr 5 :--

class MkSafeShape (s :: [Nat]) where
  mkSafeShape :: SafeShape s
instance MkSafeShape '[] where
  mkSafeShape = NilShape
instance (MkSafeShape s, KnownNat m) => MkSafeShape (m ': s) where
  mkSafeShape = Proxy :-- mkSafeShape

type family ShapeProduct (s :: [Nat]) :: Nat
type instance ShapeProduct '[] = 1
type instance ShapeProduct (m ': s) = m * ShapeProduct s

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

dConstant :: (TensorType a, ShapeProduct s ~ n) => Vector n a -> SafeShape s -> SafeTensor Build a s '[]
dConstant elems shp = SafeTensor $ constant (toShape shp) (toList elems)

dPlaceholder :: (MonadBuild m, TensorType a, KnownSymbol sym) => SafeShape s -> m (SafeTensor Value a s '[ '(sym, s)])
dPlaceholder shp = do
  pl <- placeholder (toShape shp)
  return $ SafeTensor pl

dAdd :: (TensorType a, a /= Bool)
  => SafeTensor v a s p1
  -> SafeTensor v a s p2
  -> SafeTensor Build a s (Union p1 p2)
dAdd (SafeTensor t1) (SafeTensor t2) = SafeTensor (t1 `add` t2)

dMatMul :: (TensorType a, a /= Bool, a /= Int8, a /= Int16, a /= Int64, a /= Word8, a /= ByteString)
   => SafeTensor Build a '[i,n] p -> SafeTensor Build a '[n,o] p -> SafeTensor Build a '[i,o] p
dMatMul (SafeTensor t1) (SafeTensor t2) = SafeTensor (t1 `matMul` t2)

main :: IO (VN.Vector Int64)
main = runSession $ do
  let (shape1 :: SafeShape '[2,2]) = fromJust $ fromShape (Shape [2,2])
  let (elems1 :: Vector 4 Int64) = fromJust $ fromList [1,2,3,4]
  let (elems2 :: Vector 4 Int64) = fromJust $ fromList [5,6,7,8]
  let (constant1 :: SafeTensor Build Int64 '[2,2] '[]) = dConstant elems1 shape1
  let (constant2 :: SafeTensor Build Int64 '[2,2] '[]) = dConstant elems2 shape1
  let (SafeTensor additionNode) = constant1 `dAdd` constant2 
  run additionNode

main2 :: IO (VN.Vector Float)
main2 = runSession $ do
  let (shape1 :: SafeShape '[4,3]) = fromJust $ fromShape (Shape [4,3])
  let (shape2 :: SafeShape '[3,2]) = fromJust $ fromShape (Shape [3,2])
  let (elems1 :: Vector 12 Float) = fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
  let (elems2 :: Vector 6 Float) = fromJust $ fromList [5,6,7,8,9,10]
  let (constant1 :: SafeTensor Build Float '[4,3] '[]) = dConstant elems1 shape1
  let (constant2 :: SafeTensor Build Float '[3,2] '[]) = dConstant elems2 shape2
  let (SafeTensor multNode) = constant1 `dMatMul` constant2
  run multNode

main3 :: IO Float
main3 = runSession $ do
  let (shape1 :: SafeShape '[2,2]) = fromJust $ fromShape (Shape [2,2])
  (a :: SafeTensor Value Float '[2,2] '[ '("a", '[2,2])]) <- dPlaceholder shape1
  (b :: SafeTensor Value Float '[2,2] '[ '("b", '[2,2])]) <- dPlaceholder shape1
  let result = a `dAdd` b
  runWithPlaceholders undefined result

dEncodeTensorData :: (TensorType a, TensorDataType [] a, ShapeProduct s ~ n, KnownSymbol l)
   => SafeShape s -> Vector n a -> SafeTensorData a l s
dEncodeTensorData shp elems = SafeTensorData (encodeTensorData (toShape shp) (toList elems))

-- Goal:
--
-- runWithPlaceholders :: t1 -> t2 -> t3
--   t1 = The list (or tuple) of inputs
--   t2 = The Tensor, whose type is now parameterized by exactly the input list of types we need, and who is storing the
--        nodes that will actually get attached.
--   t3 = The resulting tensor

dRun :: (TensorType a) => SafeTensor Build a s '[] -> Session a
dRun = runWithPlaceholders ()

runWithPlaceholders :: (TensorType a) => p -> SafeTensor Build a s pl -> Session a
runWithPlaceholders = undefined
