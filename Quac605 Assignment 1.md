## A.1 — State Decomposition


Let the search space have size \(N=2^n\), with \(M\) marked (target) items.
\[
|T\rangle=\frac{1}{\sqrt{M}}\sum_{x\in \text{targets}}|x\rangle,\quad
|E\rangle=\frac{1}{\sqrt{N-M}}\sum_{x\notin \text{targets}}|x\rangle.
\]
The superposition is
\[
|\psi_0\rangle=\frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle
= \sqrt{\frac{M}{N}}\,|T\rangle + \sqrt{\frac{N-M}{N}}\,|E\rangle.
\]
Let \(\sin\theta=\sqrt{\frac{M}{N}}\) and \(\cos\theta=\sqrt{\frac{N-M}{N}}\);
\[
|\psi_0\rangle=\sin\theta\,|T\rangle+\cos\theta\,|E\rangle,
\]
with \(\sin^2\theta+\cos^2\theta=1\) ensuring normalization.

--------------


