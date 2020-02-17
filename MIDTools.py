# -*- coding: utf-8 -*-
"""
@author: Guilherme Mazanti

This code implements methods for considering multiplicity-induced-dominancy
(MID) for linear delay-differential equations of retarded type with a single
delay, based on the articles:
  
[1] G. Mazanti, I. Boussaada, S.-I. Niculescu.
Multiplicity-induced-dominancy for delay-differential equations of retarded
type.
https://hal.archives-ouvertes.fr/hal-02479909

[2] G. Mazanti, I. Boussaada, S.-I. Niculescu, Y. Chitour.
Effects of roots of maximal multiplicity on the stability of some classes of
delay differential-algebraic systems: the lossless propagation case.
https://hal.archives-ouvertes.fr/hal-02463452

[3] G. Mazanti, I. Boussaada, S.-I. Niculescu, T. Vyhlídal.
Spectral dominance of complex roots for single-delay linear equations.
https://hal.archives-ouvertes.fr/hal-02422707

[4] G. Mazanti, I. Boussaada, S.-I. Niculescu.
On qualitative properties of single-delay linear retarded differential
equations: Characteristic roots of maximal multiplicity are necessarily
dominant.
https://hal.archives-ouvertes.fr/hal-02422706

TODO:
  - Implement complex conjugate roots of maximal multiplicity.
  - Extend to the case of differential-algebraic equations treated in [2].
  - Put the numerical resolution in an abstract class.
"""

import numpy as np
import scipy.special as spp
import cxroots as cx
import matplotlib.pyplot as plt
import jitcdde as jit
import scipy.integrate as sint
from scipy.interpolate import interp1d

class LinearRetardedSingleDelayDDE:
  """
  Class intended for representing a linear delay-differential equations of
  retarded type with a single delay and with a root of maximal multiplicity.
  The equation is under the form
  y^{(n)}(t) + \sum_{k=0}^{n-1} a_k y^{(k)}(t)
  + \sum_{k=0}^{n-1} \alpha_k y^{(k)}(t - \tau) = 0
  where y is the unknown function, n is the order of the equation and tau is
  the delay.
  
  Attributes:
    n: integer, order of the equation
    tau: delay
    s0: root of maximal multiplicity
    a: numpy array of shape (n,) containing the coefficients a_k
    alpha: numpy array of shape (n,) containing the coefficients alpha_k
    Delta: characteristic quasipolynomial of the equation
    DeltaPrime: derivative of Delta
  """
  def __init__(self, n, tau, s0):
    """
    n: integer, order of the equation
    tau: delay
    s0: root of maximal multiplicity
    
    The other attributes of the class are computed using the methods
    _computeCoefficients and _computeDeltaAndDeltaPrime.
    """
    self.n = n
    self.tau = tau
    self.s0 = s0
    self._computeCoefficients()
    self._computeDeltaAndDeltaPrime()

  def _computeCoefficients(self):
    """
    Computes the coefficients a_k and alpha_k ensuring that s0 is a root of
    multiplicity 2n of Delta, according to [1, 4]. These coefficients are
    stored in the attributes a and alpha of the class.
    """
    k = np.arange(self.n)
    j = np.arange(self.n)
    K, J = np.meshgrid(k, j)
    
    temp = spp.binom(J, K) * spp.binom(2*self.n - J - 1, self.n-1) \
           * (self.s0)**((J - K)*(J > K)) \
           / (spp.factorial(J) * self.tau**(self.n - J))
    self.a = spp.binom(self.n, k) * (-self.s0)**(self.n-k) \
             + (-1)**(self.n-k) * spp.factorial(self.n) * temp.sum(0)
    
    temp = (J >= K) * (-1)**((J-K)*(J>K)) * spp.factorial(2*self.n-J-1) \
           * self.s0**((J - K)*(J > K)) \
           / (spp.factorial(K) * spp.factorial((J - K)*(J > K)) \
              * spp.factorial(self.n - J - 1) * self.tau**(self.n - J))
    self.alpha = (-1)**(self.n - 1) * np.exp(self.s0 * self.tau) * temp.sum(0)

  def _computeDeltaAndDeltaPrime(self):
    """
    Computes the characteristic quasipolynomial Delta and its derivative
    DeltaPrime of the equation. Assumes that the attributes a and alpha have
    already been initialized. The quasipolynomials Delta and DeltaPrime are
    stored as attributes of the class.
    """
    k = np.arange(self.n)
    self.Delta = lambda s: s**self.n + (s**k).dot(self.a) \
                           + np.exp(-s*self.tau) * (s**k).dot(self.alpha)
    self.DeltaPrime = lambda s: self.n*s**(self.n-1)\
                                + (k[1:]*s**k[:-1]).dot(self.a[1:])\
                                + np.exp(-s*self.tau)\
                                  * ((k[1:]*s**(k[:-1])).dot(self.alpha[1:]) \
                                     - self.tau*(s**k).dot(self.alpha))
                                  
  def computeRoots(self, CoordsReal, CoordsImag):
    """
    Computes numerically the roots of the quasipolynomial Delta in a
    rectangular domain of the complex plane. The computation is carried out
    using the cxroots package.
    
    Inputs:
      CoordsReal: tuple with 2 entries
      CoordsComplex: tuple with 2 entries
      The domain on which the roots are computed is the rectangle whose 4
      vertices are
      CoordsReal[0] + 1j*CoordsComplex[0]
      CoordsReal[0] + 1j*CoordsComplex[1]
      CoordsReal[1] + 1j*CoordsComplex[0]
      CoordsReal[1] + 1j*CoordsComplex[1]
    Output:
      cxroots.RootResult.RootResult containing the roots and their
      multiplicities.
    """
    cont = cx.Rectangle(CoordsReal, CoordsImag)
    return cont.roots(self.Delta, self.DeltaPrime, M = 2*self.n+1)

  def plotRoots(self, rts, **kwargs):
    """
    Plots the roots contained in rts.
    
    Input:
      rts: cxroots.RootResult.RootResult computed using the method
        computeRoots.
      **kwargs: passed on to the scatter method of matplotlib.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([z.real for z in rts.roots], [z.imag for z in rts.roots],\
               **kwargs)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.set_title("Zeros of $\\Delta$ with $n = {}$, $\\tau = {}$,"+\
                 " and $s_0 = {}$".format(self.n, self.tau, self.s0))
    
  def integrate(self, Tfinal, Nt, initial_sol, method = "ExplicitEuler"):
    """
    Solves the differential equation numerically on the interval [0, Tfinal].
    
    Inputs:
      Tfinal: positive real number, corresponds to the final time of the
        simulation
      Nt: number of time steps in the interval [-tau, Tfinal] (for the methods
        "JitCDDE" and "AsODE") or [0, Tfinal] (for the methods "ExplicitEuler"
        and "ImplicitEuler")
      initial_sol: initial condition in [-tau, 0]. Should be given as a
        function that takes as argument a numpy array t of shape (N,)
        containing numbers in [-tau, 0] and returning a numpy array of shape
        (self.n, N).
      method (optional): method used for numerical integration (default:
        "ExplicitEuler"). Currently, four methods are implemented:
        "JitCDDE": uses the jitcdde package for performing the numerical
          integration
        "ExplicitEuler": explicit Euler method using a constant time step dt
          such that the delay is an integer multiple of dt. The final time in
          this case is not exactly Tfinal.
        "ImplicitEuler": implicit Euler method using a constant time step dt
          such that the delay is an integer multiple of dt. The final time in
          this case is not exactly Tfinal.
        "AsODE": uses the scipy.integrade package, considering delayed terms
          as a source term in the equation.
          
    Outputs:
      time: numpy array of shape (T,) containing the times at which the
        numerical solution was computed
      sol: numpy array of shape (self.n, T) containing the numerical values of
        the solution y and its derivatives of order up to n-1. sol[i, :]
        contains the approximation of y^(i).
    """
    if method=="JitCDDE":
      f = [jit.y(k) for k in range(1, self.n)]
      f.append(-self.a.dot([jit.y(k) for k in range(self.n)]) \
        - self.alpha.dot([jit.y(k, jit.t - self.tau) for k in range(self.n)]))
      equation = jit.jitcdde(f)
      equation.past_from_function(initial_sol)
      time = np.linspace(-self.tau, Tfinal, Nt)
      sol = np.zeros((self.n, time.size))
      # Do not use step_on_discontinuities: makes a HUGE error!
      #equation.step_on_discontinuities()
      equation.integrate_blindly(time[1] - time[0])
      for i in range(time.size):
        sol[:, i] = equation.integrate(time[i])
      return time, sol
    
    elif method=="ExplicitEuler":
      A0 = np.diag(np.ones(self.n - 1), 1)
      A0[-1, :] = -self.a
      
      A1 = np.zeros((self.n, self.n))
      A1[-1, :] = -self.alpha
      
      dt = Tfinal/(Nt-1)
      Npast = int(np.ceil(self.tau / dt))
      dt = self.tau / Npast
  
      time = np.arange(-self.tau, Tfinal, dt)
      sol = np.zeros((self.n, time.size))
      
      sol[:, :(Npast+1)] = initial_sol(time[:(Npast+1)])
      
      for i in range(Npast, time.size-1):
        sol[:, i+1] = sol[:, i] + dt*(A0.dot(sol[:, i]) + A1.dot(sol[:, i-Npast]))
      return time, sol
    
    elif method=="ImplicitEuler":
      A0 = np.diag(np.ones(self.n - 1), 1)
      A0[-1, :] = -self.a
      
      A1 = np.zeros((self.n, self.n))
      A1[-1, :] = -self.alpha
      
      dt = Tfinal/(Nt-1)
      Npast = int(np.ceil(self.tau / dt))
      dt = self.tau / Npast
  
      time = np.arange(-self.tau, Tfinal, dt)
      sol = np.zeros((self.n, time.size))
      
      sol[:, :(Npast+1)] = initial_sol(time[:(Npast+1)])
      
      for i in range(Npast, time.size-1):
        sol[:, i+1] = np.linalg.solve(np.eye(self.n) - dt*A0, sol[:, i] + dt*A1.dot(sol[:, i-Npast+1]))
      return time, sol
    
    elif method=="AsODE":
      A0 = np.diag(np.ones(self.n - 1), 1)
      A0[-1, :] = -self.a
      
      A1 = np.zeros((self.n, self.n))
      A1[-1, :] = -self.alpha
      
      time = np.linspace(-self.tau, Tfinal, Nt)
      sol = np.zeros((self.n, time.size))
      Npast = (time<0).sum()
      sol[:, :Npast] = initial_sol(time[:Npast])
      
      def f(t, y):
        solFunc = interp1d(time, sol, kind="cubic", copy=False)
        return A0.dot(y) + A1.dot(solFunc(t - self.tau))
      equation = sint.ode(f, lambda t, y: A0)
      equation.set_initial_value(sol[:, Npast-1])
      for i in range(Npast, time.size):
        sol[:, i] = equation.integrate(time[i])
      return time, sol
    
    else:
      raise Exception("Unknown integration method \"{}\"".format(method))
  
  def plotSolutions(self, times, sols, labels = None, leg_loc = None):
    """
    Plots the solutions computed using the integrate method.
    
    Inputs:
      times: list of numpy arrays containing times at which solutions were
        computed
      sols: list of numpy arrays with the same length as times, such that
        sols[i] contains the solution corresponding to times[i]
      labels (optional): list of strings with the same length as times, such
        that labels[i] contains a label to be used for the solution given by
        times[i] and sols[i]. If not given, no labels are shown.
      leg_loc (optional): location of the legend, in the same format of the 
        argument loc of the matplotlib method legend. If not given, default
        location is used. Ignored if labels is not given.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Solutions with $n = {}$, $\\tau = {}$, and $s_0 = {}$".format(self.n, self.tau, self.s0))
    ax.set_xlabel("Time")
    N = len(sols)
    linestyles = ["-", "--", ":", "-."]
    for i in range(N):
      if labels is not None:
        ax.plot(times[i], sols[i], linestyle = linestyles[i % 4], label = labels[i])
      else:
        ax.plot(times[i], sols[i], linestyle = linestyles[i % 4])
    if labels is not None:
      if leg_loc is not None:
        ax.legend(loc = leg_loc)
      else:
        ax.legend()

# =============================================================================
# Tests
# =============================================================================
if __name__=="__main__":
  Delta = LinearRetardedSingleDelayDDE(3, 2.5, -0.5)
  rts = Delta.computeRoots((-5, 1), (-30, 30))
  Delta.plotRoots(rts, c="black", marker="x")
  
  def initial1(t):
    y0 = np.zeros((3, t.size))
    y0[0, :] = 1
    return y0
  time1, sol1 = Delta.integrate(40, 600000, initial1)
  
  def initial2(t):
    y0 = np.zeros((3, t.size))
    y0[0, :] = -t
    y0[1, :] = -1
    return y0
  time2, sol2 = Delta.integrate(40, 600000, initial2)
  
  def initial3(t):
    y0 = np.zeros((3, t.size))
    y0[0, :] = -t**2/4
    y0[1, :] = -t/2
    y0[2, :] = -0.5
    return y0
  time3, sol3 = Delta.integrate(40, 600000, initial3)
  
  def initial4(t):
    omega = np.pi
    y0 = np.zeros((3, t.size))
    y0[0, :] = -np.sin(omega*t)/(6*omega**2)
    y0[1, :] = -np.cos(omega*t)/(6*omega)
    y0[2, :] = np.sin(omega*t)/6
    return y0
  time4, sol4 = Delta.integrate(40, 600000, initial4)
  
  Delta.plotSolutions([time1, time2, time3, time4],\
                      [sol1[0, :], sol2[0, :], sol3[0, :], sol4[0, :]],\
                      ["$y_{0, 1}$", "$y_{0, 2}$", "$y_{0, 3}$", "$y_{0, 4}$"])
  
  