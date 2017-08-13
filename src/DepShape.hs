{-# LANGUAGE GADTs               #-}
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE ScopedTypeVariables #-}

module DepShape where

import Data.Constraint (Dict)
import Data.Proxy (Proxy(..))
import Data.Vector.Sized (Vector(..))
import GHC.TypeLits (Nat, KnownNat, natVal)
import TensorFlow.Core (Shape(..), TensorType, Tensor, Build)

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

dConstant :: (TensorType a) => Vector n a -> MyShape s -> MyTensor Build a s
dConstant = undefined

