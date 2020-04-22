import streamlit as st

import math
import numpy as np
from  matplotlib import pyplot as plt
from  plotly import graph_objects as go
from typing import List
import pandas as pd

def _plot_exponentials():
    base = 1
    interval = st.slider("Interval:", min_value=0.000001, max_value=1.0, step=0.00001, value=0.5)
    t = np.arange(0., 5., interval)
    fig, ax = plt.subplots()

    fig = go.Figure()

    for b in range(base, base + 4):
        ax.plot(t, b ** t, label=f"a = {b}")
        fig.add_trace(go.Scatter(x=t, y=b**t, showlegend=True, name=f"a = {b}"))
   
    fig.update_layout(title="Plotting a^t vs t", xaxis_title="t", yaxis_title="a ^ t")
    st.plotly_chart(fig, use_container_width=True)


st.markdown(
    f"""
    # Exponentials, Derivatives and `e`

    In this article, we will observe some interesting properties of exponential functions and how the magical
    constant `e` governs all exponential functions and their derivatives.

    ## Exponential Functions

    To refresh your memory, exponential functions are typically expressed as follows.
    let `f(t)` be an exponential function of `t` that operates on some constant number `a`:
    """
)

st.latex(
    r"""
    f(t) = a ^ t
    """
)


st.markdown(
    """
    Let's start out by plotting some familiar exponential values.
    """
)


_plot_exponentials()


st.markdown("""
    ## Derivatives of Exponential Functions

    Recall that the expression for computing the derivative of any function `f(t)` is:
""")

st.latex(
    r"""
    f'(t) = \frac{df}{dt} = \lim_{dt \to 0} \frac{f(t + dt) - f(t)}{dt}
                          = \frac{f(t + dt) - f(t)}{dt} \hspace{3 pt} \left( \text {where dt is small enough.} \right)
    """
)

st.markdown("""
    Let us substitute the exponential expression:
""")

st.latex(
r'''
    f'(t) = \frac{df}{dt} = \frac{a^{t + dt} - a ^ t}{dt} \hspace{3 pt} \left( \text {where dt is small enough.} \right)
''')

st.markdown("""
    Using elementary algebra, we can rewrite the expression as follows:
""")

st.latex(
r'''
    f'(t) = \frac{df}{dt} = a^t \left( \frac{a^{dt} - 1}{dt} \right)
''')


st.markdown("""
    In other words, the derivative of an exponential function at `t` is equal to the value of the exponential function
    at `t` multiplied by some number `c`.
    
    Note that for a given value of `a` and for a small enough constant value
    of `dt`, `c` is also a constant value.

    >**This is a key insight into the innate property of exponential functions that the rate of change of an exponential
    >function is proportional to the value of the exponential function.**

    Now, let's observe the values of this multiplier `c` for different values of `a`: 
""")

def _plot_exponential_derivative() -> None:
    interval = st.slider("Interval:", min_value=0.000001, max_value=1.0, step=0.00001, value=0.5, key='i2')
    dt = st.number_input(
        "Choose `dt` for computing derivative:",
        min_value=0.00001, max_value=0.1, value=0.001, step=0.05
    )
    t = np.arange(0., 10., interval)

    a = st.number_input(
        "Choose a value of `a` as the base of the exponential function (example: 2.0):",
        min_value=0.01, max_value=100.0, value=2.0, step=0.5
    )

    fig, ax = plt.subplots()
    ax.plot(t, a ** t, 'r+', label="f(t)")
    ax.plot(
        t, (a ** (t + dt) - a ** t) / dt, 'bo', label="f'(t)",
    )

    _ = ax.legend()
    ax.set_xlabel("t")
    fig.suptitle(f"Base: {a}, c: {((a ** dt) - 1) / dt}")
    st.pyplot(fig, use_container_width=True)
    # st.plotly_chart(fig, use_container_width=True)

_plot_exponential_derivative()

st.markdown("""
    > **Before moving to the next section, can you adjust the `a` so that the scaling
    > factor between the derivative and the exponential is (almost) `1`?**

    Hint: If you play with the `a` number for a bit, you will see that the scaling
    factor is close to `1.0` when `a` is approximately `2.72`.  Keep this number in mind,
    as it will appear once again later in this tutorial.
    """
)

st.markdown("""
    ## Magical Multiplier to Rule All Exponential Derivatives
    Now, let's approach the same setup from a different perspective.

    Recall from the earlier sections that `f'(t) = f(t) * c` where:
""")

st.latex(r'c = \left(\frac{a^{dt} - 1}{dt}\right) \text{for some small value of dt.}')

st.markdown("""
    To gain more insight into the behavior of this multiplier, we will observe its values
    for various values of `a`. For convention, let's rewrite it as a function in `a`:
""")
st.latex(r'g(a) = \left(\frac{a^{dt} - 1}{dt}\right) \text{for some small value of dt.}')

def _plot_multiplier_vs_base() -> None:
    interval = st.slider(
        "Interval:", min_value=0.000001, max_value=1.0, step=0.00001, value=0.5, key='i4'
    )
    bases = np.arange(1., 50., interval)

    dt = st.number_input(
        "Choose `dt` for computing derivative:",
        min_value=0.00001, max_value=0.1, value=0.001, step=0.05, key='dt1'
    )

    fig, ax = plt.subplots()
    ax.plot(bases, (bases ** dt - 1) / dt, '+')
    ax.set_xlabel('a')
    ax.set_ylabel('g(a)')
    st.plotly_chart(fig, use_container_width=True)

    g = [
        [2, (2 ** dt - 1) / dt],
        [4, (4 ** dt - 1) / dt],
        [5, (5 ** dt - 1) / dt],
        [8, (8 ** dt - 1) / dt],
        [25, (25 ** dt - 1) / dt],
        [64, (64 ** dt - 1) / dt],
    ]
    st.dataframe(pd.DataFrame.from_records(g, columns=['a', 'g(a)']))

_plot_multiplier_vs_base()

st.markdown("""
    Notice the peculiar shape of this chart.
    To gain more insight in the shape, observe the hand-picked values in the table.
    
    From the above table, you can see that:
""")

st.latex(r'''
    g(a^2) = 2 g(a)
''')

st.markdown("""
    This is a strong hint that `g(a)` is a logarithmic function!
    But it is not quite the [common logarithm](https://en.wikipedia.org/wiki/Common_logarithm),
    because we know that `log` of `2` to the base `10` is approximately `0.30103`.

    **So what is really the base of this logarithmic function?**

    ### Finding the Base of of Logarithmic Multiplier
    Let `x` be the unknown base of the logarithm function represented by `g`:
""")

st.latex(r'''
    g(a) = log_x (a) \hspace{2 pt} \text {where x represents the unknown base of this log function.}
''')

st.write("Let's take the specific examples from the table above:")
st.latex(r"""log_x(2) = 0.6931 \dots""")
st.latex(r"""log_x(4) = 1.3873 \dots""")
st.latex(r"""log_x(5) = 1.6107 \dots""")
st.latex(r"""log_x(8) = 2.0816 \dots""")
st.latex(r"""log_x(16) = 2.7764 \dots""")
st.latex(r"""log_x(25) = 3.2241 \dots""")
st.latex(r"""log_x(64) = 4.1675 \dots""")

st.markdown("""
    To empirically find `x` in the above expression, let's just plot the expression for
    various values of `x`.
""")

def _plot_exponents_for_x():
    interval = st.slider(
        "Interval:", min_value=0.000001, max_value=1.0, step=0.00001, value=0.5, key='i5'
    )
    x = np.arange(1., 10., interval)
    
    a = st.number_input(
        "Choose a value of `a` for the exponential (example: 2.0):",
        min_value=0.01, max_value=10.0, value=2.0, step=1.0
    )

    fig, ax = plt.subplots()
    ax.plot(x, np.log(a) / np.log(x), '+')
    ax.set_xlabel('x')
    ax.set_ylabel('log(a) to the base x')
    fig.suptitle(f"a: {a}")
    st.plotly_chart(fig, use_container_width=True)

_plot_exponents_for_x()

st.markdown(f"""
    In this chart, try varying the value of `a`, and for each value of `a`, try to find
    the value of `x` that corresponds to a y-axis value that matches `log_x(a)` in the table above.

    For example, if we set `a` to `2.0`, what is the value of `x` that is close to the value
    `0.6931...` on the y-axis?  The answer is a number between `2.7` and `2.8`, which is
    suspiciously close to the number we observed earlier.

    As another example, if we set `a` to `5.0`, what is the value of `x` that is close to the value
    `0.6931...` on the y-axis?  The answer is again a number between `2.7` and `2.8`, which is
    suspiciously close to the number we observed earlier.

    This in fact is the magical number `e` that is the base of the logarithmic function represented by `g`!
""")

st.latex(r"""g(a) = log_e(a) = ln(a)""")

st.markdown("""
    >The more precise value of `e` according to the `numpy` package is {np.e}.

    ## Connecting the Dots
    This magical number `e` underpins the phenomenon where the rate of change of an exponential function `a^t` at
    a certain value of `t` is proportional to the value `a^t` multiplied by the log of `a` to the base `e`:
""")

st.latex(r"""f(t) = a^t""")
st.latex(r"""
    f'(t) = a^t log_e(a) = a^t ln(a)
""")

st.markdown("""
    The natural extension of this expression is when `a = e`, `f'(t) = f(t)`:
""")

st.latex(r"""f(t) = e^t""")
st.latex(r"""
    f'(t) = e^t log_e(e) = e^t
""")


st.markdown("""
    ## References

    This tutorial is basically a watered down version of this mindblowing [YouTube video](https://www.youtube.com/watch?v=m2MIpDrF7Es).
    
    I created this tutorial just to convince myself of all the math in the video and also to play with Streamlit open-source library.
    Do check out the documentation at [https://docs.streamlit.io/](https://docs.streamlit.io/) and follow along interesting discussions
    at [https://discuss.streamlit.io/](https://discuss.streamlit.io/).

""")