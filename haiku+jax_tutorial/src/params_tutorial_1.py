from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp



class Container(hk.Module):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self) -> None:
        
        m = hk.get_parameter("m", (5,2), init=hk.initializers.Constant(7))


def foo() -> None:
    c = hk.get_parameter("c", (5,2), init=hk.initializers.Constant(5))
    d = hk.get_parameter("d", (5,2), init=hk.initializers.Constant(9)) 
    container = Container()
    container()


foo_transformed = hk.transform(foo) #not foo(). foo() is a return value of function foo


key = jax.random.PRNGKey(23)
'''

    RNG:Random Number Generator


    *args
    関数に渡す位置引数をまとめて受け取る記法です。
    python
    def f(*args):
        print(args)  # タプルになる

    f(1, 2, 3)  # → (1, 2, 3)
    foo_transformed.init(rng, *args)の意味は「RNGキーの後ろに、元の関数fooに渡す引数をいくつでも続けられる」ということです。
'''

params = foo_transformed.init(key) # foo_transformedは，Trandformed型で，initというメソッドを持つ
breakpoint()


