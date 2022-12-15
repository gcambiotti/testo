#!/usr/bin/env python
# coding: utf-8

# # Setup

# <div class="alert alert-info">
# 
# **Note:** 
#     
# This is a note!
# 
# </div>

# In[1]:


import ipywidgets as widgets

a = widgets.IntSlider(
    value=7,
    min=0,
    max=10,
    step=1,
    description='Test:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
    )


# In[2]:


a


# In[3]:


a.value


# In[4]:


import plotly.express as px
data = px.data.iris()
data.head()


# In[5]:


import altair as alt
alt.Chart(data=data).mark_point().encode(
    x="sepal_width",
    y="sepal_length",
    color="species",
    size='sepal_length'
)


# In[6]:


import plotly.io as pio
import plotly.express as px
import plotly.offline as py

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", size="sepal_length")
fig


# $$
#   w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
# $$ (my_other_label)

# ```{math}
# :label: my_label
# w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
# ```

# - A link to an equation directive: {eq}`my_label`
# - A link to a dollar math block: {eq}`my_other_label`

# :::{dropdown}
# This text is **standard** _Markdown_
# :::

# :::{dropdown}
# This text is **standard** _Markdown_
# :::
# 
# 
# ```{dropdown} Here's my dropdown
# And here's my dropdown content
# ```
# 
# ```{note}
# :class: dropdown
# The note body will be hidden!
# ```
# 
# 
# ```{admonition} Click here!
# :class: tip, dropdown
# Here's what's inside!
# ```
# 
# ````{admonition} Click here!
# :class: tip, dropdown
# Here's what's inside!
# ````
# 
# ````{tip} Click here!
# Here's what's inside!
# ````
# 

# <div class="alert alert-block alert-danger"><b>Danger:</b> This alert box indicates a dangerous or potentially negative action.</div>

# In[7]:


0


# In[8]:


0


# In[9]:


get_ipython().run_line_magic('matplotlib', 'widget')
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

from obspy.clients.fdsn import Client
client = Client("INGV")
print(client)

from obspy import UTCDateTime
starttime = UTCDateTime(2018,8,1)
endtime   = UTCDateTime(2018,8,31)

rad_km,lon,lat = [660,13,42]

earth_radius = 6371      # [km]
rad_deg = (rad_km / earth_radius) * 180/np.pi

catalog = client.get_events(starttime=starttime,endtime=endtime,           # time interval
                            latitude=lat,longitude=lon,maxradius=rad_deg,  # circle
                            minmagnitude=3.5,orderby="magnitude")            # minimum magnitude
print(catalog)


# In[10]:


import folium

m = folium.Map(
    location=[lat,lon],width="70%",
    zoom_start=5)

folium.Circle(radius=rad_km*1e3,location=[43,13],color='#3388ff',fill=True).add_to(m)

for event in catalog:
    origin = event.preferred_origin()
    mag = event.preferred_magnitude().mag
    #description = event.descriptions[0].text
    folium.Marker(
        location=[origin.latitude,origin.longitude],
        popup='Mw = {1:.1f} Z = {0:.1f} km'.format(origin.depth/1e3,mag),
        icon=folium.Icon(icon='star')
    ).add_to(m)

m


# With these commands we allow interactive figures (`%matplotlib widget`) and we make available the `matplotlib.pyplot`and `numpy` modules with the aliases `plt` and `np`. We also set the way in which the numpy array will be printed and define variable for $2\,\pi$.

# In[11]:


a=5


# In[12]:


get_ipython().run_line_magic('run', 'outher.ipynb')


# In[13]:


c


# In[14]:


get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)
pi2 = 2*np.pi


# # Dirac delta function

# ### Definition

# The Dirac delta can be loosely thought of as a function on the real line which is zero everywhere except at the origin, where it is infinite,
# 
# $$
# \delta(x) = \begin{cases} \infty & x=0 \\ 0 & x\neq 0 \end{cases}
# $$
# 
# and which is also constrained to satisfy the identity
# 
# $$
# \int_{-\infty}^\infty \delta(x)\,\mathrm{d}x = 1
# $$
# 
# A better characterization consists in consdering the Dirac delta function as a generalized function. In the theory of distributions, a generalized function is considered not a function in itself but only about how it affects other functions when "integrated" against them. In keeping with this philosophy, to define the delta function properly, it is enough to say what the "integral" of the delta function is against a sufficiently "good" test function $f$
# 
# $$
# \int_{\mathcal{I}} f(x)\,\delta(x)\,\mathrm{d}x = 
# \begin{cases} f(0) & 0\in\mathcal{I} \\ 
# 0 & 0\notin\mathcal{I}
# \end{cases}
# $$
# 
# where $\mathcal{I}$ is an open interval of the real axis. In this respect, the Dirac delta can be defined as the distributional derivative of the Heaviside step function
# 
# $$
# \delta(x) = H'(x) 
# $$
# 
# where the prime $'$ stands for the first-order derivative and $H$ is the Heaviside function
# 
# $$
# H(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x<0 \end{cases}
# $$
# 
# 
# ```{admonition} Proof of eq. (1)
# :class: tip, dropdown
# 
# Let us consider the integration by parts of two function $f(x)$ and $g(x)$
# 
# $$
# \int_a^b f(x)\,g'(x)\,\mathrm{d}x = f(x)\,g(x)\,\Big|_a^b - \int_a^b f'(x)\,g(x)\,\mathrm{d}x
# $$
#     
# and apply it to the case in which $g(x)=H(x)$ and $a<0<b$
# 
# $$
# \begin{align}
# \int_a^b f(x)\,\delta(x)\,\mathrm{d}x &= f(x)\,H(x)\,\Big|_a^b - \int_a^b f'(x)\,H(x)\,\mathrm{d}x \\
# &= f(b) - \int_0^b f'(x)\,\mathrm{d}x = f(b) - f(x)\,\Big|_0^b = f(0)
# \end{align}
# $$
# 
# ```

# ### Dirac delta derivative

# The Dirac delta derivative is defined as follows
# 
# $$
# \begin{align}
# \int_a^b f(x)\,\delta'(x)\,\mathrm{d}x &= f(x)\,\delta(x)\,\Big|_a^b - \int_a^b f'(x)\,\delta(x)\,\mathrm{d}x \\
# &=  -f'(0)
# \end{align}
# $$
# 
# For higher-order derivatives one obtians
# 
# $$
# \begin{align}
# \int_a^b f(x)\,\delta^{(n)}(x)\,\mathrm{d}x &= (-1)^n\,f^{(n)}(0)
# \end{align}
# $$

# ### Properties of the Dirac delta

# #### Basic propetries 
# 
# The Dirac delta is symmetric
# 
# $$
# \begin{align}
# \delta(-x) = \delta(x)
# \end{align}
# $$
# 
# and the product of the Dirac delta with $x$ is equal to zero
# 
# $$
# \begin{align}
# x\,\delta(x) = 0
# \end{align}
# $$
# 
# The Dirac delta derivative satisfies a number of basic properties
# 
# $$
# \begin{align}
# & \delta'(-x) = -\delta'(x) \\
# & x\,\delta'(x) = -\delta(x)
# \end{align}
# $$

# # Fourier transform

# ### Definition

# The Fourier transform of a function $f(t)$ in the time domain is defined as
# 
# \begin{equation*}
# \tilde{f}(\omega) = \int_{-\infty}^\infty f(t)\,e^{-i\,\omega\,t}\,\mathrm{d} t
# \tag{1.1a}
# \end{equation*}
# 
# where $t$ is the time, $\omega=2\,\pi\,f$ is the angular frequency, with $f$ being the frequency, and $i$ is the imaginary unit.
# 
# 
# We can also define the inverse of the Fourier transform, through which we can obtain the function in the time domain starting from its Fourier transform
# 
# \begin{equation*}
# f(t)  = \frac{1}{2\,\pi} \int_{-\infty}^\infty \tilde{f}(\omega)\,e^{i\,\omega\,t}\,\mathrm{d} \omega
# \label{eq:1.1b}\tag{1.1b}
# \end{equation*}
# 
# The above definition allows us to better understand the meaning of the Fourier transform. Indeed, we can see the function in the time domain as the superimposition of monochromaic signals (with given frequencies) where the Fourier transform provides the weights of each monochromatic signal. 
# 
# <div class="alert alert-block alert-success"><b><u>Dirac delta function:</u></b> 
# 
# Let us consider the Dirac delta function $\delta(t)$ and its Fourier transform
# 
# \begin{equation*}
# \tilde{\delta}(\omega) = \int_{-\infty}^\infty \delta(t)\,e^{-i\,\omega\,t}\,\mathrm{d} t = 1
# \end{equation*}
# 
# From \eqref{eq:1.1b}, we obtain
# 
# \begin{equation*}
# \delta(t) = \frac{1}{2\,\pi}\,\int_{-\infty}^\infty 1\,e^{i\,\omega\,t}\,\mathrm{d} \omega
# \end{equation*}
# 
# </div>
# 
# 
# <div class="alert alert-block alert-success"><b><u>Constant function:</u></b> 
# 
# Let us consider the constant function $c(t)$ 
#     
# \begin{equation}
#     c(t) = 1
#     \end{equation}
#     
# and evaluate its Fourier transform
# 
# \begin{equation*}
# \tilde{c}(\omega) = \int_{-\infty}^\infty 1\,e^{-i\,\omega\,t}\,\mathrm{d} t = 2\,\pi\,\delta(\omega)
# \label{eq:fou_constant}\tag{1.2}
# \end{equation*}
# 
# where we have made use of $\delta(\omega) = \delta(-\omega)$.
#     
# </div>
# 
# 
# <div class="alert alert-block alert-success">
# <b><u>Box function:</u></b> 
#     
# Let us calculate the Fourier transform of the box function $B_T(t)$ which yields one in the interval $[0,T)$ and zero elsewhere
# 
# \begin{equation*}
# B_T(t) = H(t)-H(t-T) =\begin{cases} 1 & t\in[0,T) \\ 0 & t\notin [0,T)
# \end{cases}
# \label{eq:1.2}\tag{1.2}
# \end{equation*}
#         
# with $H4 being the Heaviside step function, eq. (). In this case we have
#     
# \begin{align}
# \tilde{B}_T(\omega) &= \int_{-\infty}^\infty B_T(t)\,e^{-i\,\omega\,t}\,\mathrm{d} t = \int_0^T e^{-i\,\omega\,t}\,\mathrm{d} t = \frac{e^{-i\,\omega\,t}}{-i\,\omega}\bigg|_0^T \\
#                                              &= \frac{1-e^{-i\,\omega\,T}}{i\,\omega}
# \label{eq:1.3}\tag{1.3}
# \end{align}
#     
# <div class="alert alert-block alert-info">
# <b>Tip</b> 
#     
# Despite the denominator in eq. \eqref{eq:1.3}, $\tilde{B}(\omega)$ is continuous at $\omega=0$ as shown hereinafter
#     
# \begin{equation*}
# \lim_{\omega \rightarrow 0} \tilde{B}_T(\omega) = \lim_{\omega \rightarrow 0} \frac{1-(1-i\,\omega\,T)}{i\,\omega} = T
# \end{equation*}
#     
# In order to avoid any numerical issue in evaluating this Fourier transform at $\omega=0$, it is convinient to implement the following form
#     
# 
# \begin{align}
# \tilde{B}_T(\omega) &= e^{-i \frac{\omega\,T}{2}}\, \frac{2\,\sin\big(\frac{\omega\,T}{2}\big)}{\omega} \\
#     &= T\,e^{-i\,\frac{\omega\,T}{2}}\,j_0\big(\tfrac{\omega\,T}{2}\big)
# \end{align}
# 
#     
# where $j_0$ is the spherical Bessel function of the first kind and order 0
#     
# \begin{equation*}
# j_0(x) = \frac{\sin(x)}{x}
# \end{equation*} 
# 
# </div>
# </div>

# ### Properties of the Fourier transform

# #### Shift in time
# 
# The Fourier transform of a function shifted in time $f(t+t_0)$ yields the Fourier transform $\tilde{f}(\omega)$ multiplied by $e^{i\,\omega\,t_0}$
# 
# \begin{align}
# \int_{-\infty}^\infty f(t+t_0)\,e^{-i\,\omega\,t}\,\mathrm{d}t 
# = e^{i\,\omega\,t_0}\,\tilde{f}(\omega)
# \end{align}

# #### Derivatives
# 
# The Fourier transform of a $n$-th order time derivative of a function $f(t)$ yields the Fourier transform $\tilde{f}(\omega)$ multiplied by $(i\,\omega)^n$
# 
# \begin{align}
# \int_{-\infty}^\infty \frac{\mathrm{d}^n f(t)}{\mathrm{d} t^n}\,e^{-i\,\omega\,t}\,\mathrm{d}t = (i\,\omega)^n\,\tilde{f}(\omega)
# \end{align}

# #### Convolution
# 
# The Fourier transform of the convolution between two functions $f(t)$ and $g(t)$ 
# 
# \begin{equation}
# h(t) = \int_{-\infty}^\infty f(\tau)\,g(t-\tau)\,\mathrm{d}\tau 
# \end{equation}
# 
# yields the product of the two Fourier transforms $\tilde{f}(\omega)$ and $\tilde{f}(\omega)$ 
# 
# \begin{align}
# \int_{-\infty}^\infty h(t)\,e^{-i\,\omega\,t}\,\mathrm{d}t = \tilde{f}(\omega)\,\tilde{g}(\omega)
# \end{align}

# #### Real functions
# 
# The Fourier transform of a real function $f(t)$ has the following property
# 
# \begin{equation}
# \tilde{f}^*(\omega) = \tilde{f}(-\omega)
# \end{equation}
# 
# We thus can write its inverse Fourier transform as follows
# 
# \begin{equation}
# f(t) = \frac{1}{\pi}\,\Re\left[\int_0^\infty \tilde{f}(\omega)\,e^{i\,\omega\,t}\right]\,\mathrm{d} \omega = \frac{1}{\pi}\,\int_0^\infty \Big(\Re\big[\tilde{f}(\omega)\big]\,\cos(\omega\,t) - \Im\big[\tilde{f}(\omega)\big]\,\sin(\omega\,t) \Big)\,\mathrm{d} \omega
# \end{equation}
# 
# where $\Re$ and $\Im$ are the operators which returns the real and imaginary parts.

# ### Plotting the Spherical Bessel function 

# Let us make our first plot of the spherical Bessel function of order $n=0$

# In[15]:


from scipy.special import spherical_jn                      # import the spherical Bessel functions

xs = np.linspace(-10,10,10001)                              # numpy array of the evenly space grid of the interval [-10,10]
j0s = spherical_jn(0,xs)                                    # numpy array of the sample of the spherical Bessel function of order n=0

fig,ax = plt.subplots(1,tight_layout=True,figsize=(7,3))    # setting a figure with one panel
ax.plot(xs,j0s);                                            # plotting the spherical Bessel function on the panel
ax.axhline(0,color="black",linewidth=0.5);                  # draw the horizontal line


# ### Plotting the box function and its Fourier transfrom

# Let us implement ([1.2](#mjx-eqn-eq:1.2))  and ([1.4](#mjx-eqn-eq:1.4)) in the following two python functions `Box` and `FBox`. Note that both functions have the optional parameter `T=1` which provide the length of the time interval $[0,T)$ of the box function

# In[16]:


def Box(ts,T=1):
    
    ys = np.ones(ts.shape)
    ys[ts<0] = 0
    ys[ts>=T] = 0
    
    return ys

def FBox(fs,T=1):
    
    ws = pi2 * fs
    xs = ws * T / 2
    Ys = T * np.exp(-1j*xs) * spherical_jn(0,xs)
    
    return Ys


# Then, we make a choice for the value of $A$ (`A=4`), we generate two evenly space grids in the time and frequency domains and we calculate the box function and its Fourier transform over the grids

# In[17]:


T = 4

ts = np.linspace(-1,T+1,10001)
print("ts =",ts)

fs = np.linspace(-5/T,5/T,10001)
print("fs =",fs)

ys = Box(ts,T)
Ys = FBox(fs,T)


# In the end, we set a figure with two panels and plot the samplings of the two functions

# In[18]:


fig,axes = plt.subplots(1,2,tight_layout=True,figsize=(9,4));     # setting a figure (`fig`) with two panels (àxes`)

axes[0].plot(ts,ys);
axes[0].set_xlabel("Time [s]")
axes[0].set_title("Box function")

axes[1].plot(fs,Ys.real,label="Re");
axes[1].plot(fs,Ys.imag,label="Im");
axes[1].legend();
axes[1].set_xlabel("Frequency [Hz]");
axes[1].set_title("Fourier transform");

for ax in axes:
    ax.axhline(0,color="black",linewidth=0.5);
axes[1].axvline(0,color="black",linewidth=0.5);


# We note that the Fourier transform is a complex function and, so, we plot both its real and imaginary parts. Also, we can see that the real part is symmetric with respect to the origin, while the imaginary part is antisymmetric. This is because the box function is a real function.

# # Periodic function and the Fourier series 

# ### Definition

# Let us consider a periodic function $f(t)$ with period $T$
# 
# \begin{equation}
# f(t+n\,T) = f(t)  \qquad\mathrm{with}\quad n\in\mathbb{N}
# \end{equation}
# 
# Such a function can be expanded in Fourier series as follows
# 
# \begin{equation}
# f(t) = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,e^{i\,\omega_k\,t}
# \end{equation}
# 
# where $\omega_k$ are the following discretized frequencies
# 
# \begin{align}
# & \omega_k = k\,\delta\omega \qquad\mathrm{with}\quad \delta \omega = \frac{2\,\pi}{T}\quad\mathrm{and}\quad k\in\mathbb{N}
# \end{align}
# 
# and $F_k$ are the Fourier coefficients given by
# 
# \begin{equation}
# F_k = \int_0^T f(t)\,e^{-i\,\omega_k\,t}\,\mathrm{d} t \qquad\mathrm{with}\quad k\in\mathbb{N}
# \end{equation}
# 
# Its Fourier transform yields
# 
# \begin{equation}
# \tilde{f}(\omega) = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,\int_{-\infty}^\infty e^{i\,(\omega_k-\omega)\,t}\,\mathrm{d} t = \frac{2\,\pi}{T}\,\sum_{k=-\infty}^\infty F_k\,\delta(\omega-\omega_k)
# \label{eq:2.1}\tag{2.1}
# \end{equation}
# 
# and, so, it can be seen as the superimposition of a infinite number of monochromatic signals with discretized frequencies $\omega_k$ (with $k\in\mathbb{N}$).
# 
# 
# <div class="alert alert-block alert-success"><b><u>Constant function:</u></b> 
# 
# The constant function $f(t)=1$ can be seen as a periodic function for any period $T$. In this case, we have 
#     
#     
# \begin{equation}
#     F_k = \int_0^T 1\,e^{-i\,\omega_k\,t}\,\mathrm{d} t = \begin{cases} T & k=0 \\ 0 & k\neq 0\end{cases}
#     \end{equation}
#     
# and \eqref{eq:2.1} becomes 
# 
# \begin{equation*}
# \tilde{f}(\omega) = 2\,\pi\,\delta(\omega)
# \tag{1.1a}
# \end{equation*}
# 
# which corresponds to eq. ([1.2](#mjx-eqn-eq:fouconstant))
# </div>
# 
# 

# ### Compact support

# Let us consider a function $f$ with compact support on the interval $[0,T)$. This means that we are assuming that the function is equal to zero outside the compact support
# 
# \begin{equation}
# f(t) = 0 \qquad\mathrm{for}\quad t\notin [0,T)
# \end{equation}
# 
# Thanks to the finiteness of the compact support, we can represent the function using the Fourier series as follows
# 
# \begin{equation}
# f(t) = \frac{B_T(t)}{T}\,\sum_{k=-\infty}^\infty F_k\,e^{i\,\omega_k\,t} 
# \end{equation}
# 
# where the Fourier coefficients are still given by eq. (). We note that we have introduced the box function over the time interval $[0,T)$ to guarantee eq. ().
# 
# The Fourier transform of the function with compact support yields
# 
# \begin{align}
# \tilde{f}(\omega) &= \int_{-\infty}^\infty f(t)\,e^{-i\,\omega\,t}\,\mathrm{d} t = \int_0^T f(t)\,e^{-i\,\omega\,t}\,\mathrm{d} t = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,\int_0^T e^{i\,(\omega_k-\omega)\,t}\,\mathrm{d} t = \frac{1}{T}\,\sum_{k=-\infty}^\infty F_k\,\frac{e^{i\,(\omega_k-\omega)\,T}-1}{i\,(\omega_k-\omega)} \\
# &=\sum_{k=-\infty}^\infty F_k\,e^{i\frac{(\omega_k-\omega)\,T}{2}}\,j_0\big(\tfrac{(\omega_k-\omega)\,T}{2}\big)
# \end{align}
# 
# <div class="alert alert-block alert-success"><b><u>The link between the Fourier series and transform for functions with compact support:</u></b> 
# 
# By considering that 
#     
# \begin{align}
#     & j_0(k\,\pi)=0 \qquad \forall\,k\in\mathbb{N} 
# \end{align}
# 
# we note that the Fourier coefficients $F_k$ corresponds to the Fourier transform of the function with compact support evaluated at the discretized frequencies $\omega_k$
# 
# \begin{align}
# \tilde{f}(\omega_k) &= F_k
# \end{align}
# 
# </div>

# Let us now enlarge the compact support to time intervals $[0,A)$ with $A>T$. In this perspective, we rewrite eqs. () and () for this enlarged compact support
# 
# \begin{align}
# & f(t) = \frac{B_A(t)}{A}\,\sum_{q=-\infty}^\infty F_q'\,e^{i\,\omega_q'\,t}  \\
# & \tilde{f}(\omega) = \sum_{q=-\infty}^\infty F_q'\,e^{i\frac{(\omega_q'-\omega)\,A}{2}}\,j_0\big(\tfrac{(\omega_q'-\omega)\,A}{2}\big)
# \end{align}
# 
# where
# 
# \begin{align}
# & \omega_q' = q\,\delta\omega' \quad\mathrm{and}\quad \delta\omega' = \frac{2\,\pi}{A}
# \end{align}
# 
# and note that eq. () applies even to the present case
# 
# \begin{align}
# \tilde{f}(\omega_q') &= F_q'
# \end{align}
# 
# In light of this and of eq. (), the Fourier coefficients $F_q'$ for the compact support $[0,A)$ can be obtained from those of the compact support $[0,T)$ as follows
# 
# \begin{align}
# F_q' = \sum_{k=-\infty}^\infty F_k\,e^{i\frac{(\omega_k-\omega_q')\,T}{2}}\,j_0\big(\tfrac{(\omega_k-\omega_q')\,T}{2}\big)
# \end{align}
# 

# ### Exercise 1

# Let us consider the following function with compact support
#     
# \begin{equation}
# f(t) = \big[1 - \cos(2\,\pi\,t/T)\big]\,B_T(t)
# \end{equation}
# 
# We will refer to this function as the *compact wave* function.
# 
# #### Task:
# 
# 1. Calculate the Fourier coefficients of the Fourier series over the intervals $[0,T)$ and $[0,A)$, with $A>T$.
# 1. Calculate the Fourier transform.
# 1. Plot the function and its Fourier transform.

# #### Solution (1)

# Let us rewrite eq. as follows
# 
# \begin{equation} f(t) = \left(1 -\frac{e^{i\,\frac{2\,\pi\,t}{T}} + e^{-i\,\frac{2\,\pi\,t}{T}}}{2}\right)\,B_T(t) 
# \end{equation}
# 
# with $\delta\omega = \frac{2\,\pi}{T}$. From the comparison with eq. (), we can write
# 
# \begin{equation}
# f(t) = \frac{B_T(t)}{T}\,\sum_{k=-\infty}^\infty F_k\,e^{i\,\omega_k\,t}
# \end{equation}
# 
# with
# 
# \begin{equation}
# F_k = \begin{cases} T & k=0 \\ -T/2 & k=\pm1 \\ 0 &k\neq0,\pm1 \end{cases}\end{equation}

# #### Solution (2)

# From eqs. () and (), we obtain
# 
# \begin{equation}
# \tilde{f}(\omega) = \frac{T}{2}\,\bigg(2\,e^{-i\,\omega\,T/2}\,j_0\big(\omega\,T/2\big)-e^{-i\,(\delta\omega-\omega)\,T/2}\,j_0\big((\delta\omega-\omega)\,T/2\big)+e^{i\,(\delta\omega+\omega)\,T/2}\,j_0\big((\delta\omega+\omega)\,T/2\big)\bigg)
# \end{equation}

# #### Solution (3)

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

def Wave(ts,T=1):
    
    ys = 1 - np.cos(pi2*ts/T)
    ys *= Box(ts,T)

    return ys

def FWave(fs,T=1):
    
    ws = pi2*fs
    
    dw = pi2/T
    xp = (dw-ws)*T/2
    xm = (dw+ws)*T/2
    
    ys = T/2 * ( 2*np.exp(-1j*ws*T/2)*spherical_jn(0,ws*T/2) - np.exp(-1j*xp)*spherical_jn(0,xp) - np.exp(-1j*xm)*spherical_jn(0,xm) ) 

    return ys

ts = np.linspace(-1,T+1,10001)
fs = np.linspace(-10/T,10/T,10001)

ys = Wave(ts)
Ys = FWave(fs)

fig,axes=plt.subplots(2,tight_layout=True,figsize=(7,6))
axes[0].plot(ts,ys)
axes[1].plot(fs,Ys.real)
axes[1].plot(fs,Ys.imag)

for ax in axes:
    ax.axhline(0,color="black",linewidth=0.5)



# # The Discrete Fourier transform and its inverse

# ### Definition

# Let us consider a function $g(t)$ and its sampling over an evenly spaced grid in the interval $[0,A)$
# 
# \begin{equation}
# \mathbf{g}=[g_0,\cdots,g_{n-1}]
# \end{equation}
#  
# where
#  
# \begin{align}
# &g_j = g(t_j)  \qquad\mathrm{with}\quad j=0,\cdots,n-1
# \end{align}
# 
# with
# 
# \begin{align}
# & t_j = j\,\delta t   \qquad\mathrm{with}\quad\delta t = \frac{A}{n} \quad\mathrm{and}\quad j=0,\cdots,n-1
# \end{align}
# 
# The discrete Fourier transform operator $\mathrm{DFT}$ to the sampling $\mathbf{g}$ yields 
# 
# \begin{equation}
# \hat{\mathbf{G}} = \mathrm{DFT}\,[\mathbf{g}] = \big[\hat{G}_1,\cdots,\hat{G}_{n-1}\big]
# \end{equation}
# 
# where $\hat{G}_k$ are the following complex coefficients
# 
# \begin{align}
# & \hat{G}_k = \delta t\,\sum_{j=0}^{n-1} f_j\,e^{-i\,\omega_k\,t_j} \qquad\mathrm{with} \quad k=0,\cdots,n-1
# \end{align}
# 
# with
# 
# \begin{align}
# & \omega_k = k\,\delta \omega   \qquad\mathrm{with}\quad\delta \omega = \frac{2\,\pi}{A}\quad\mathrm{and}\quad k=0,\cdots,n-1 
# \end{align}
# 
# We can also define the inverse discrete Fourier transform operator $\mathrm{IDFT}$ as follows
# 
# \begin{equation}
# \mathbf{g} = \mathrm{IDFT}\big[\hat{\mathbf{G}}\big] = [g_0,\cdots,g_{n-1}]
# \end{equation}
# 
# where
# 
# \begin{equation}
# g_j = \frac{1}{T}\,\sum_{k=0}^{n-1} \hat{G}_k\,e^{i\,\omega_k\,t_j}
# \end{equation}
# 
# It allows to obtain the samples $\mathbf{g}$ starting from the series of coefficients of the discrete Fourier transform $\hat{\mathbf{G}}$. This can be proved by inserting eq. () into eq. () and making use of the summation of finite geometric series
# 
# \begin{equation}
# \sum_{j=0}^{n-1} r^j = \frac{1-r^n}{1-r}
# \end{equation}
# 
# For the sake of the example, let us now consider the box function $B_T(t)$ and sample it over an evenly spaced grid of the interval $[0,A)$ with time step $\delta t$, with $A>T$. We choose $T=1.3\,\mathrm{s}$, $A=2\,\mathrm{s}$ and $\delta t=0.25\,\mathrm{s}$ and we check that the IDFT of the DFT corresponds to the sampling of $B_T(t)$

# In[20]:


T,A,dt = 1.3,2,0.4
n = int(A/dt)
p = n//2
dt = A/n

ts_n = np.linspace(0,A,n,endpoint=False)
ys_n = Wave(ts_n,T=T)

Ys_n = np.fft.fft(ys_n) * dt

ys_n_check = np.fft.ifft(Ys_n).real / dt

print("Sample of the box function =",ys_n)
print("Difference                 =",ys_n-ys_n_check)


# ### Mathematical application

# According to eq. (), the series of the DFT $\hat{\mathbf{G}}=[\hat{G}_0,\cdots,\hat{G}_{n-1}]$ is associated to the following series of frequencies $\boldsymbol{\omega} = [\omega_0,\cdots,\omega_{k-1}]$. On the other hand, in the perspective of using the DFT for approximating the Fourier transform of the function $g(t)$ and for the case in which $n=2\,p$ is even, it is possible to recast eq. () as follows
# 
# \begin{equation}
# g_j = \sum_{k=0}^{p-1} G_k\,e^{i\,\omega_k\,t} + \sum_{k=-p}^{-1} \hat{G}_{n-k}\,e^{i\,\omega_k\,t} \qquad\mathrm{with}\quad n=2\,p
# \end{equation}
# 
# and, so, associate the series of the DFT $\hat{\mathbf{G}}=[\hat{G}_0,\cdots,\hat{G}_{p-1},\hat{G}_p,\cdots,\hat{G}_{n-1}]$ to the following series of frequencies $\boldsymbol{\omega} = [\omega_0,\cdots,\omega_{p-1},\omega_{-p},\cdots,\omega_{-1}]$, where the first and last $p$ frequencies are non-negative e negative, respectively.
# 
# When $n=2\,p+1$ is odd, instead, eq. () can be recast as follows
# 
# \begin{equation}
# g_j = \sum_{k=0}^{p} G_k\,e^{i\,\omega_k\,t} + \sum_{k=-p}^{-1} \hat{G}_{n-k}\,e^{i\,\omega_k\,t} \qquad\mathrm{with}\quad n=2\,p-1
# \end{equation}
# 
# and, so, associate the series of the DFT $\hat{\mathbf{G}}=[\hat{G}_0,\cdots,\hat{G}_{p},\hat{G}_p,\cdots,\hat{G}_{n-1}]$ to the following series of frequencies $\boldsymbol{\omega} = [\omega_0,\cdots,\omega_{p},\omega_{-p},\cdots,\omega_{-1}]$, where the first $p+1$ and last $p$ frequencies are non-negative e negative, respectively.
# 
# The Nyquist frequency
# 
# \begin{equation}
# f_{\mathrm{Ny}} = \frac{p}{A}
# \end{equation}
# 
# is the maximum frequency (in absolute value) that contributes to the the DFT.
# 
# As one can check, the function `np.fft.fftfreq` returns the above series of frequencies, taking into account both the even and odd cases of $n$ 

# In[21]:


T, A, dt = 1.3, 2, 0.4

n = int(A/dt)                                    # number of samples
dt = A/n                                         # recalculate the time step
p = n//2
if n % 2 == 1: p += 1

ts_n = np.linspace(0,A,n,endpoint=False)

fs_n = np.fft.fftfreq(n,d=dt)
Fny = p/A

print("n,p               = ",n,p)
print("Times             = ",ts_n)
print("Nyquist frequency = ",Fny)
print("Frequency         = ",fs_n)
print("Non-negative      = ",fs_n[:p])
print("Negative          = ",fs_n[p:])


# In the perspective of simplfying the plotting of the DFT, we develop a python function which sort the frequencies (or the series of the DFT coefficients) with increasing frequencies

# In[22]:


def Sort(fs):
    n = len(fs)
    p = n//2
    if n % 2 == 1: p += 1
    return np.append(fs[p:],fs[:p])


# Let us now plot the sampling of the box function over the interval $[0,A)$ and its DFT choosing $T=1.2\,\mathrm{s}$, $A=4\,\mathrm{s}$ and $\delta t=0.2 \,\mathrm{s}$ and check that the IDFT returns the same value of the original sampling in the time domain

# In[23]:


T, A, dt = 1.3, 2, 0.4

n = int(A/dt)                                    # number of samples
dt = A/n                                         # recalculate the time step
p = n//2

ts_n = np.linspace(0,A,n,endpoint=False)         # evenly spaced grid of the interval [0,A)
fs_n = np.fft.fftfreq(n,d=dt)                    # associated discrete frequencies

ys_n = Wave(ts_n,T=T)                             # sampling of the box function over ts_n
Ys_n = np.fft.fft(ys_n) * dt                     # DFT of the sampling ys_n
check_ys_n = np.fft.ifft(Ys_n).real / dt         # IDFT of the series of coefficients Ys_n

fs_n = Sort(fs_n)
Ys_n = Sort(Ys_n)


fig,axes = plt.subplots(1,3,tight_layout=True,figsize=(11,4))

line_option = dict(marker=".",markersize=2,color="red",linewidth=0.5)

axes[0].plot(ts_n,ys_n,marker="o",label="sampling")
axes[0].plot(ts_n,check_ys_n,marker=".",label="check")
axes[0].axvline(T,color="black",linewidth=0.5)
axes[0].legend()
axes[0].set_xlabel("Time [s]")
axes[0].set_title("Sampling in time domain")

axes[1].plot(fs_n,Ys_n.real,marker=".")
axes[1].set_title("Real part")
axes[1].set_xlabel("Frequency [Hz]")

axes[2].plot(fs_n,Ys_n.imag,marker=".")
axes[2].set_title("Imaginary part")
axes[2].set_xlabel("Frequency [Hz]")

for ax in axes:
    ax.axhline(0,color="black",linewidth=0.5)
    ax.axvline(0,color="black",linewidth=0.5)


# Let us now compare the above approach with the original box function and its Fourier transform. In this perspetive, we define the following python function where we can set the parameters $T$, $A$ and $\delta t$ as optional arguments 

# In[24]:


#####################################################
def Plot(T=1.3,A=2,dt=0.4):
    
    n = int(A/dt)
    dt = A/n

    ts_n = np.linspace(0,A,n,endpoint=False)         # evenly spaced grid of the interval [0,A)
    fs_n = np.fft.fftfreq(n,d=dt)                    # associated discrete frequencies

    ys_n = Wave(ts_n,T=T)
    Ys_n = np.fft.fft(ys_n) * dt                     # DFT of the sampling ys_n

    Fny = (n//2)/A

    ts = np.linspace(-0.5,A+0.5,10001)
    fs = np.linspace(-1.2*Fny,1.2*Fny,10001)

    ys = Wave(ts,T=T)
    Ys = FWave(fs,T=T)
    
    fs_n = Sort(fs_n)
    Ys_n = Sort(Ys_n)

    fig,axes = plt.subplots(1,3,tight_layout=True,figsize=(12,4))

    line_option = dict(marker=".",markersize=2,color="red",linewidth=0.5)
    
    axes[0].plot(ts,ys,label="continuos")
    axes[0].plot(ts_n,ys_n,**line_option,label="discrete")
    axes[0].axvline(T,color="black",linewidth=0.5)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_title("Sampling in time domain")

    axes[1].plot(fs,Ys.real,label="continuos")
    axes[1].plot(fs_n,Ys_n.real,**line_option,label="discrete")
    axes[1].set_title("Real part")
    axes[1].set_xlabel("Frequency [Hz]")

    axes[2].plot(fs,Ys.imag,label="continuos")
    axes[2].plot(fs_n,Ys_n.imag,**line_option,label="discrete")
    axes[2].set_title("Imaginary part")
    axes[2].set_xlabel("Frequency [Hz]")

    for ax in axes:
        ax.legend()
        ax.axhline(0,color="black",linewidth=0.5)
        ax.axvline(0,color="black",linewidth=0.5)
    
    axes[0].axvspan(0,A,alpha=0.1)
    for ax in axes[1:]:
        ax.axvspan(-Fny,Fny,alpha=0.1)
        
    return (fig,axes)
#####################################################


# and we make use of it

# In[25]:


fig,axes = Plot()


# Increasing the number of samples $n$ we enlarge the range of frequencies that we can explore

# In[26]:


fig,axes = Plot(dt=0.2)


# Increasing the time window $[0,A)$, instead, we refine the discretization of the frequencies

# In[27]:


fig,axes = Plot(A=8,dt=0.2)


# So far, the size of the time window was too small and the time step was too large. Choosing very large time windows and fine time steps, the DFT well approximate the analytical Fourier transform of the box function

# In[28]:


fig,axes = Plot(A=20,dt=0.02)


# Let us zoom the previous figure in order to better appreciate the goodness of the approximation

# In[29]:


axes[0].set_xlim(-0.5,2.5)
for ax in axes[1:]:
    ax.set_xlim(-2.5,2.5)
fig.show()


# ### How the approximation works

# In order to understand why the approximation works, we have to think that when we sample the function in the finite time window $[0,A)$, we miss all the information before and after the time interval. Let us consider a function $g(t)$ with a compact support within the time window $[0,A)$
# 
# \begin{equation}
# g(t) = 0  \qquad\mathrm{with}\quad t\notin [0,A)
# \end{equation}
# 
# Then, let us define a periodic function $h$ with period $A$ defined as
# 
# \begin{equation}
# h(t) = \sum_{m=-\infty}^\infty g(t+m\,A)
# \end{equation}
# 
# so that 
# 
# \begin{align}
# & h(t+m\,A) =h(t) \\
# & h(t) = g(t) \quad \mathrm{with}\quad t\in [0,A)
# \end{align}
# 
# Obviously, we have
# 
# \begin{equation}
# g(t) = h(t)\,B_A(t)
# \end{equation}
# 
# Let us now expand the periodic function $h(t)$ in the Fourier series
# 
# \begin{equation}
# h(t) = \frac{1}{A}\,\sum_{k=-\infty}^\infty H_k\,e^{i\,\omega_k\,t}
# \end{equation}
# 
# where
# 
# \begin{align}
# & H_k = \int_0^A h(t)\,e^{-i\,\omega_k\,t}\,\mathrm{d} t = \int_0^A g(t)\,e^{-i\,\omega_k\,t}\,\mathrm{d} t \qquad \mathrm{with}\quad k\in\mathbb{N} \\
# & \omega_k = k\,\delta\omega \qquad\mathrm{with}\quad \delta\omega=\frac{2\,\pi}{A}\quad\mathrm{and}\quad k\in\mathbb{N}
# \end{align}
# 
# In light of eq. (), the Fourier coefficients $H_k$ corresponds to the Fourier transform of $g(t)$ evaluated at the discretized frequencies $\omega_k$
# 
# \begin{equation}
# \tilde{g}(\omega_k) = H_k
# \end{equation}
# 
# In this respect, increasing the length of the time window $A$, we can refine the frequency step $\delta\omega$ and obatin the Fourier transform over finer and finer grids in the frequency domain. The probelm that arises in the numerical approach is that we can sample the function over a finite numbers of times and, so, the integral must be approximated in some way. 
# 
# Let us assume that we know the function $g(t)$ only at the times $t_j=j\,\delta t$ with $\delta t=A/n$ and $j=0,\cdots,n-1$. In this case, the simplest approximation consists in
# 
# \begin{align}
# H_k &= \int_0^A g(t)\,e^{-i\,\omega_k\,t}\,\mathrm{d} t \approx \delta t\,\sum_{j=0}^{n-1} g_j\,e^{-i\,\omega_k\,t_j} \\
# &= \delta t\,\sum_{j=0}^{n-1} g_j\,e^{-i\,\omega_k\,t_j} = \hat{G}_k
# \end{align}
# 
# where $\hat{G}_k$ are the approximated Fourier coefficients. We note that they correspond to the complex coefficients of the DFT given by eq. (), although the latter are limited to $k=0,\cdots,n-1$. Also, from eq. (), we known that we need only the first $n$ complex complex coefficients $\hat{H}_k$ to perform the IDFT
# 
# \begin{equation}
# g(t_j) = g_j = \frac{1}{A}\,\sum_j \hat{G}_k\,e^{i\,\omega_k\,t_j}
# \end{equation}
# 
# An alternative understanding of the approximation, consists in noting that $\omega_{k+m\,n}\,t_j=\omega_k\,t_j+2\,\pi\,m$ for $k=0,\cdots,n-1$ and $m\in\mathbb{N}$. In this respect, evaluating eq. () at the discretized times $t_j$, we can write
# 
# \begin{equation}
# h(t_j) = \frac{1}{A}\, \sum_{k=0}^{n-1}\left(\sum_{m=-\infty}^\infty H_{k+m\,n}\right)\,e^{i\,\omega_k\,t_j}
# \end{equation}
# 

# In[ ]:




