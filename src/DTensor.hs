{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module DTensor where

import Data.Int (Int64)
import TensorFlow.Core
import TensorFlow.Ops (add, constant)

import Data.Singletons.Prelude
import GHC.TypeLits

-- newtype (TensorType a) => DTensor v a (s :: Nat) = DTensor { unTensor :: (Tensor v a) }

data DTensor v a s where
  DTensor :: (TensorType a) => Tensor v a -> DTensor v a s
  -- DTensorFull :: Tensor v a -> DTensor v a (x ': xs) 

data MyShape (s :: [Nat]) where
  NilShape :: MyShape '[]
  (:--) :: Sing m -> MyShape s -> MyShape (m ': s)

dAdd :: (TensorType a, a /= Bool) => DTensor v a s -> DTensor v a s -> DTensor Build a s
dAdd (DTensor t1) (DTensor t2) = DTensor (t1 `add` t2)

data Natu = Z | S Natu

data SNatu n where
  SZ :: SNatu 'Z
  SS :: SNatu n -> SNatu ('S n)

data Vector a n where
  Nil :: Vector a 'Z
  (:-) :: a -> Vector a n -> Vector a ('S n)
infixr :-

fromList :: SNatu n -> [a] -> Maybe (Vector a n)
fromList SZ _ = Just Nil
fromList (SS n) (x : xs) = (x :-) <$> fromList n xs
fromList _ _ = Nothing

{-fromShape :: (l ~ '[]) => SList l -> Shape -> Maybe (MyShape n l)
fromShape SNil _ = Just NilShape
fromShape (SCons h ts)  (Shape (a : as)) = (h :--) <$> fromShape ts as
fromShape _ _ = Nothing-}

fromShape :: Shape -> MyShape s
fromShape (Shape []) = NilShape
fromShape (Shape (a : as)) = (sing :: Sing a) :-- fromShape (Shape as)

{-fromShape :: SList l -> MyShape n l
fromShape SNil = NilShape
fromShape (SCons h ts) = h :-- fromShape ts-}
-- Will this even help? Can I get here from Shape? 
-- What other constraints do I need to add to make this type check and work out?


-- We want the shape to be inferred, except it can't necessarily be inferred just from the
-- list. We need the shape as well.
-- dConstant :: (TensorType a, HasShape s) => s -> [a] -> DTensor Build a (MyShapeFamily s)
--dConstant shp as = DTensor (constant (getShape shp) as)

-- :: (TensorType a, ShapeMatches x y) => x -> y -> DTensor Build a (ShapeOf x)

-- 1. Can I attach the exact shape that is given as the type list?
-- 2. Can I then prove that the given shape matches the number of elements in the list?

-- import Data.Vector (Vector)
-- import TensorFlow.Core (Tensor, Build)

{-data DTensor :: [Nat] -> * where
  BFTensor :: (Tensor Build Float) -> DTensor a

-- Goal: Replace final `[Int]` with a type that is constrained somehow by the input shape, and
-- which puts some constraint on the vector itself.
dConstant :: [Int] -> Vector a -> DTensor b
dConstant = undefined-}

{-data Nat = Z | S Nat

infixl 6 :+
infixl 7 :*

type family (n :: Nat) :+ (m :: Nat) :: Nat
type instance 'Z :+ m = m
type instance ('S n) :+ m = 'S (n :+ m)

type family (n :: Nat) :* (m :: Nat) :: Nat
type instance 'Z :* _ = 'Z
type instance ('S n) :* m = (n :* m) :+ m

data SNat n where
  SZ :: SNat 'Z
  SS :: SNat n -> SNat ('S n)

data BinTree a where
  BTNil :: BinTree a
  BTNode :: a -> BinTree a -> BinTree a -> BinTree a

data SBinTree t where
  SBTNil :: SBinTree 'BTNil
  SBTNode :: a -> SBinTree t1 -> SBinTree t2 -> SBinTree ('BTNode a t1 t2)

data Vector a n where
  Nil :: Vector a 'Z
  (:-) :: a -> Vector a n -> Vector a ('S n)
infixr :-

deriving instance Eq a => Eq (Vector a n)

toList :: Vector a n -> [a]
toList Nil = []
toList (x :- xs) = x : toList xs

instance Show a => Show (Vector a n) where
  showsPrec d = showsPrec d . toList

head_ :: Vector a ('S n) -> a
head_ (x :- _) = x

tail_ :: Vector a ('S n) -> Vector a n
tail_ (_ :- xs) = xs

{-fromList :: [a] -> Vector a n
fromList [] = Nil
fromList (a : as) = a :- (fromList as)-}

append :: Vector a n -> Vector a m -> Vector a (n :+ m)
append Nil ys = ys
append (x :- xs) ys = x :- append xs ys

main :: IO ()
main = do
  print $ head_ ((1 :: Integer) :- 2 :- Nil)
  print $ tail_ ((1 :: Integer) :- 2 :- Nil)
  -- print $ head_ Nil

replicate_ :: SNat n -> a -> Vector a n
replicate_ SZ _ = Nil
replicate_ (SS n) a = a :- replicate_ n a

infixl 6 %:+

(%:+) :: SNat n -> SNat m -> SNat (n :+ m)
SZ %:+ m = m
SS n %:+ m = SS (n %:+ m)-}
