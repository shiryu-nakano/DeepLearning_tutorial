from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp

#def foo() -> None:


#c = hk.get_parameter("c", (5,2), init=hk.initializers.Constant(5))
'''
ValueError: `hk.get_parameter` must be used as part of an `hk.transform`
'''
# 
#def foo() -> None:
#    c = hk.get_parameter("c", (5,2), init=hk.initializers.Constant(5))
#foo()

'''
nkn4ryu@MacBook-Pro ~/ghq/github.com/shiryu-nakano/DeepLearning_tutorial/haiku+jax_tutorial/src %
uv run python -W ignore hello.py
Traceback (most recent call last):
  File "/Users/nkn4ryu/ghq/github.com/shiryu-nakano/DeepLearning_tutorial/haiku+jax_tutorial/src/hello.py", line 18, in <module>
    foo()
  File "/Users/nkn4ryu/ghq/github.com/shiryu-nakano/DeepLearning_tutorial/haiku+jax_tutorial/src/hello.py", line 16, in foo
    c = hk.get_parameter("c", (5,2), init=hk.initializers.Constant(5))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/nkn4ryu/ghq/github.com/shiryu-nakano/DeepLearning_tutorial/haiku+jax_tutorial/.venv/lib/python3.12/site-packages/haiku/_src/base.py", line 631, in get_parameter
    assert_context("get_parameter")
  File "/Users/nkn4ryu/ghq/github.com/shiryu-nakano/DeepLearning_tutorial/haiku+jax_tutorial/.venv/lib/python3.12/site-packages/haiku/_src/base.py", line 367, in assert_context
    raise ValueError(
ValueError: `hk.get_parameter` must be used as part of an `hk.transform`
'''
def foo() -> None:
    c = hk.get_parameter("c", (5,2), init=hk.initializers.Constant(5))

foo_transformed = hk.transform(foo) #not foo(). foo() is a return value of function foo
print(type(foo_transformed)) #<class 'dict'>
# <class 'haiku._src.transform.Transformed'>
'''
hk.transformは，callable（関数オブジェクトそのもの）を引数にもつ．
デコレータ系のAPIに渡す時は，callableを渡す

'''


key = jax.random.PRNGKey(23)
params = foo_transformed.init(key) # foo_transformedは，Trandformed型で，initというメソッドを持つ
print(type(params))
breakpoint()
'''
(Pdb) pp params
{'~': {'c': Array([[5., 5.],
       [5., 5.],
       [5., 5.],
       [5., 5.],
       [5., 5.]], dtype=float32)}}
'''
print(params) 
# {モジュール名: {パラメータ名: 配列}}
'''
{'~': {'c': Array([[5., 5.],
       [5., 5.],
       [5., 5.],
       [5., 5.],
       [5., 5.]], dtype=float32)}}
nkn4ryu@MacBook-Pro ~/ghq/github.com/s
'''

# params = foo_transformed.init(rng, *args, **kwargs)
'''
RNGとは？
*argsとは
**kwargsとは？


'''
