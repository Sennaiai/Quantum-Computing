## Quantum Computing Essentials

### 1. Linear Algebra  
**Inner product** – multiplication of two vectors resulting in a scalar; measures the overlap between the vectors.
  
**Outer product** – combines two vectors to form a matrix.
  
**Vector space** – a set of vectors.

**Orthonormal** - a set of normalized and orthogonal vectors.
  
**Tensor product** – a process of joining vector spaces into a larger joint space; represents composite quantum states,                      expanding the Hilbert space.
  
**Eigen vector** - non-zero vector, whose direction remains unchanged under a given linear tasnformation
  
**Eigen value** - a scalar factor that represents a linear transformation of an eigen-vector.
  
---

### 2. Quantum Mechanics  
**Wave function interpretation** – a mathematical description of the quantum state of a particle, expressing the probability amplitudes of all possible states.  

**Quantum state** – the complete description of a system’s physical (observable or measurable) conditions.  

**Dirac notation** – mathematical notation representing a quantum state.  

**Probability amplitude** – a complex number whose magnitude squared gives the probability of a specific quantum outcome.  

**Hilbert space** – a complete vector space used to describe all possible quantum states.  

**Bloch sphere** – geometric representation of a quantum state.  

**Basis states** – a set of fundamental quantum states spanning a Hilbert space; any state can be expressed as a linear combination of the basis states.  

    Orthonormal reference states. Eg |1> and |0>
    
**Measurement** – an interactive process with a quantum system that causes it to collapse into one definite outcome among all possible states represented by its wave function. 

**Expectation value** – the average outcome of measuring an observable on a quantum state.  

**Operators** – mathematical objects that act on quantum states to produce new measurable states; represent observables, evolutions, or transformations of quantum states.  

**Hermitian operators** – represent observable quantum states.  

**Unitary operators** – represent reversible quantum evolutions of a given quantum state.  

**Uncertainty principle** – a principle defining the limitation on certain observables that cannot be precisely determined simultaneously.  

**Commutators** – mathematical operation measuring how two operators fail to be measured simultaneously.

**Superpositions** - a linear composition of linear basis states representing a system existing in multiple possible states simultaneously until measuremnt collapses it into one definite state.

**Entanglement** - a unique quantum state shared between two or more states that exist in sync with one another regardless of distancea. 

**Bell-state** - a maximally entangled quantum state of two qubits, where the systems remain perfectly correlated.

**Qubit** - a basic unit of single qauntum infomration.

**Initialization** - process of preparing a qubit in a known starting state.

**Eigen-state** - qauntum state with a definite value for a given observable.
                - definite value is the eigen value and the obsrvable is the operator.
                
**Observable** - a measurable physical quantity like energy, momentum, position ...etc
---

### 3. Density Matrix

**Matrix represenation** - a way of mathematical expression of quantum states and operators as matrices and vectors

**Density matrix** - mathematical representation of a quantum system's state. Useful for describing mixed states.

**Pure state** - completley known qauntum state that can be represented by a single state vector in Hilbert space

**Mixed state** - a statistical representation of a mixture of multiple pure states, describing a system that is not in one definite state but exists in a probability distribution over several possible states.

**Ensemble** - the physical or conceptual collection of quantum states, statisticaly represented by mixed state.

---

### 4. Gates

 **Gate** - basic operation in quantum computing that chnages the stae of a single or multiple qubits.

 **Phase** - change of amplitude or orientation of the quantum state.
         
         Changes the way qubits interact or interefere with one another

 **Rotation** - change of the actual position of quanutm state of a qubit
         
         **Changes the weight or probability of a given state**
 
**Single gate** - acts on single qubit at a time 

    Pauli X - Bit flip (like NOT gate)
    Pauli Z - Phase flip 
    Pauli Y - Bit and Phase flip 
    R(x/y/z) - Rotation around x/y/z
    Hadamard - Creates superposition 

**Multi-qubit gates** act on multiple qubits at a time 
   
    CX - Controlled X (Conditional NOT)
    CZ - Conditional Z (phase flip)
    CCX (Toffolli) - 2x Conditional NOT (Similar to CCNOT in classical) 
    SWAP - exchanges the qubit states of 2 qubits at a time

**Combination** - more gates can be created by combining the above single and/or multi qubit gates.
---

### 5. Circuits

**No-Cloning thereom** - A qauntum state can not be completle copied or cloned, due to the linearity of the state.
       
         Cloning is non-lnear whereas Quanutm is linear.

**Teleportation** - the trasnfer of qauntum state from one entangled pair to its other entangled pair instantaneously regardelss to distance. 
    
    Requires exchage of classical infomration, as infomration can not travel faster than light.
    Happy Einstein! 

**Deustch** - a quantum algorithm that demonstrates quantum parallelism, capable of determining whether a given function is constant or balanced in a single evaluation.

**Deustch-Josza** - Generalization of Deutsch, determines if a given function of multiple input bits is constant or balanced in a single shot.

**QFT** - quantum analogue of classical discrete fourier transform, and is used to transform quantum states in computational basis into frequency basis.
    
    Frequency → encoded in phase → affects amplitude → determines measurement probability.

**Phase Estimation** - an algorithm that estimates a phase associated with a given quantum state.

      Phase is the eigen-value associated with the eigenstate of a given operator.
      Core algorithm enabling multiple applications (Shors, simulation, Chemistry) 

**Quantum Parallelism** - ability of quantum computer to evaluate multiple inputs of a given function simultaneously through superposition.

    All **f(x)* values are evaluated on the Amplitudes of the quantum state all at once. 
---

### 6. Schroedinger equation

**Stationary states** - Are time independent in measurement outcomes. Although their phase evolves, their probabilty distrubitions do not chnage. 

    Eigenstates of the Hamiltonian , with eigenvalue E equivalent to the Energy of the quantum system.

**Hamiltonian** - is the energy operator of a quanutm systems representing the total energy of the system.

**Energy** - is the quantum observable corresponding to the Hamiltonian, and its eigenvalues represent the allowed energy levels of a system.

**Energy levels** - the discrete values of energy that a quantum state can possess when measured. 
    
    The discrete values of energy are the eigenvalues of the Hamiltonian operator eigenstate, and each corresponds to the eigenstate of the system. Combination of all possible eigenstates of the system forms the complete set of basis states spanning the system's Hilbert space, whereby any possible state of the system can be represented as a superposition of these eigenstates.

**PE** - represents the influence of internal or external forces acting on a system (or particle) 

**KE** - represents the energy of a system (or particle) related to motion, and depends on the momentum operator. 

      Hamiltonian combines both KE and PE to give the total energy of the system. 

**Infinite square well potential** - a model where an elctron is trapped in a well. 

    Its potential energy is zero inside the box and infinite outside
    Its total energy is equal to its kinetic energy, and is qauntized due to the boundary confinement. 
    Demonstrates the effect of boundry conditions in quantum mechanics on quantization of energy.

**Heisenberg picture** - formalism of quantum mchanics equivalent to Schroedinger's 

      Operators evolve in time while states remain static, opposite to Schrodinger picture, where states evolve and operators stay fixed.

**Uncertainty principle** - Quantum structure and nature limits measurement precision. 

      The more precisely you know a particle’s position, the less precisely you can know its momentum — and vice versa
      As such, certain pairs of physical properties can not be measured exactly and simultaneously. 
---

### 7. Evolution 

**Time evolution** - describes how a quantum state changes with time according to the Schrodinger equation. Is determined by the system’s Hamiltonian.

**Ising model** - statistical model used for optimaizaion problems. 

    Is a mathematical model used to desribe interacting spin systems 

**Trotter's rule** - a way of approximating the time evolution of a complex hamiltonian.

    Breaks the complex operation into simpler, sequential opertions that can be implemented on a quantum computer.





  

 
