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
import           Data.Int (Int64, Int8, Int16)
import           Data.Maybe (fromJust)
import           Data.Proxy (Proxy(..))
import           Data.Type.List (Union)
import qualified Data.Vector as VN
import           Data.Vector.Sized (Vector, toList, fromList)
import           Data.Word (Word8)
import           GHC.TypeLits (Nat, KnownNat, natVal)
import           GHC.TypeLits

import           TensorFlow.Core
import           TensorFlow.Core (Shape(..), TensorType, Tensor, Build)
import           TensorFlow.Ops (constant, add, matMul, placeholder)
import           TensorFlow.Session (runSession, run)
import           TensorFlow.Tensor (TensorKind)

instance Eq Shape where
  (==) (Shape s) (Shape r) = s == r

data SafeShape (s :: [Nat]) where
  NilShape :: SafeShape '[]
  (:--) :: KnownNat m => Proxy m -> SafeShape s -> SafeShape (m ': s)

data SafeTensor v a (s :: [Nat]) (p :: [Symbol]) where
  SafeTensor :: (TensorType a) => Tensor v a -> SafeTensor v a s p

data SafeTensorData a (n :: Symbol) (s :: [Nat]) where
  SafeTensorData :: (TensorType a) => TensorData a -> SafeTensorData a n s

data SafeFeed where
  SafeFeed :: SafeTensor v a s p -> SafeTensorData a n s -> SafeFeed

data FeedList (ss :: [Symbol]) where
  EmptyFeedList :: FeedList '[]
  (:--:) :: (KnownSymbol n) => (SafeTensor Value a s p, SafeTensorData a n s) -> FeedList ss -> FeedList (n ': ss)

infixr 5 :--
infixr 5 :--:

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

safeConstant :: (TensorType a, ShapeProduct s ~ n) => Vector n a -> SafeShape s -> SafeTensor Build a s '[]
safeConstant elems shp = SafeTensor (constant (toShape shp) (toList elems))

safePlaceholder :: (MonadBuild m, TensorType a, KnownSymbol sym) => SafeShape s -> m (SafeTensor Value a s '[sym])
safePlaceholder shp = do
  pl <- placeholder (toShape shp)
  return $ SafeTensor pl

safeAdd :: (TensorType a, a /= Bool, TensorKind v)
  => SafeTensor v a s p1
  -> SafeTensor v a s p2
  -> SafeTensor Build a s (Union p1 p2)
safeAdd (SafeTensor t1) (SafeTensor t2) = SafeTensor (t1 `add` t2)

safeMatMul :: (TensorType a, a /= Bool, a /= Int8, a /= Int16, a /= Int64, a /= Word8, a /= ByteString, TensorKind v)
   => SafeTensor v a '[i,n] p1 -> SafeTensor v a '[n,o] p2 -> SafeTensor Build a '[i,o] (Union p1 p2)
safeMatMul (SafeTensor t1) (SafeTensor t2) = SafeTensor (t1 `matMul` t2)

main :: IO (VN.Vector Int64)
main = runSession $ do
  let (shape1 :: SafeShape '[2,2]) = fromJust $ fromShape (Shape [2,2])
  let (elems1 :: Vector 4 Int64) = fromJust $ fromList [1,2,3,4]
  let (elems2 :: Vector 4 Int64) = fromJust $ fromList [5,6,7,8]
  let (constant1 :: SafeTensor Build Int64 '[2,2] '[]) = safeConstant elems1 shape1
  let (constant2 :: SafeTensor Build Int64 '[2,2] '[]) = safeConstant elems2 shape1
  let (SafeTensor additionNode) = constant1 `safeAdd` constant2 
  run additionNode

main2 :: IO (VN.Vector Float)
main2 = runSession $ do
  let (shape1 :: SafeShape '[4,3]) = fromJust $ fromShape (Shape [4,3])
  let (shape2 :: SafeShape '[3,2]) = fromJust $ fromShape (Shape [3,2])
  let (elems1 :: Vector 12 Float) = fromJust $ fromList [1,2,3,4,1,2,3,4,1,2,3,4]
  let (elems2 :: Vector 6 Float) = fromJust $ fromList [5,6,7,8,9,10]
  let (constant1 :: SafeTensor Build Float '[4,3] '[]) = safeConstant elems1 shape1
  let (constant2 :: SafeTensor Build Float '[3,2] '[]) = safeConstant elems2 shape2
  let multNode = constant1 `safeMatMul` constant2
  safeRun_ multNode

main3 :: IO (VN.Vector Float)
main3 = runSession $ do
  let (shape1 :: SafeShape '[2,2]) = fromJust $ fromShape (Shape [2,2])
  (a :: SafeTensor Value Float '[2,2] '["a"]) <- safePlaceholder shape1
  (b :: SafeTensor Value Float '[2,2] '["b"] ) <- safePlaceholder shape1
  let result = a `safeAdd` b
  (result_ :: SafeTensor Value Float '[2,2] '["b", "a"]) <- safeRender result
  let (feedA :: Vector 4 Float) = fromJust $ fromList [1,2,3,4]
  let (feedB :: Vector 4 Float) = fromJust $ fromList [5,6,7,8]
  let fullFeedList = (b, safeEncodeTensorData shape1 feedB) :--:
                     (a, safeEncodeTensorData shape1 feedA) :--:
                     EmptyFeedList
  safeRun fullFeedList result_

safeEncodeTensorData :: (TensorType a, TensorDataType VN.Vector a, ShapeProduct s ~ n, KnownSymbol l)
   => SafeShape s -> Vector n a -> SafeTensorData a l s
safeEncodeTensorData shp elems = SafeTensorData (encodeTensorData (toShape shp) (VN.fromList (toList elems)))

-- Goal:
--
-- runWithPlaceholders :: t1 -> t2 -> t3
--   t1 = The list (or tuple) of inputs
--   t2 = The Tensor, whose type is now parameterized by exactly the input list of types we need, and who is storing the
--        nodes that will actually get attached.
--   t3 = The resulting tensor

safeRender :: (MonadBuild m) => SafeTensor Build a s pl -> m (SafeTensor Value a s pl)
safeRender (SafeTensor t1) = do
  t2 <- render t1
  return $ SafeTensor t2

safeRun_ :: (TensorType a, Fetchable (Tensor v a) r) => SafeTensor v a s '[] -> Session r
safeRun_ = safeRun EmptyFeedList

safeRun :: (TensorType a, Fetchable (Tensor v a) r) =>
  FeedList ss -> SafeTensor v a s ss -> Session r
safeRun feeds (SafeTensor finalTensor) = runWithFeeds (buildFeedList feeds []) finalTensor
  where
    buildFeedList :: FeedList ss -> [Feed] -> [Feed]
    buildFeedList EmptyFeedList accum = accum
    buildFeedList ((SafeTensor tensor_, SafeTensorData data_) :--: rest) accum = buildFeedList rest ((feed tensor_ data_) : accum)
