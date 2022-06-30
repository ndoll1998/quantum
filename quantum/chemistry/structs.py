import numpy as np
from .orbital import GaussianOrbital
from copy import deepcopy
from itertools import chain
from typing import Union, List

class Atom(object):
    """ Atom described by a set of GTOs with shared origin

        Args:
            basis (List[GaussianOrbital]): list of GTOs describing the electron orbitals of the atom
            Z (int): principal quantum number, i.e. charge of the atom
    """

    def __init__(
        self,
        basis:List[GaussianOrbital],
        Z:int
    ) -> int:
        # save copy of basis and charge
        self.basis = deepcopy(basis)
        self.Z = Z
        # check basis
        assert len(self.basis) > 0, "No GTOs provided!"
        assert all((o.origin == basis[0].origin).all() for o in basis[1:]), "Origins don't match!"
    
    @property
    def origin(self) -> np.ndarray:
        """ Origin of the atom """
        return self.basis[0].origin

    @origin.setter
    def origin(self, origin:np.ndarray) -> None:
        # convert to numpy array
        origin = np.asarray(origin)
        # update origin of all GTOs
        for o in self.basis:
            o.origin = origin

    def __add__(self, other:Union["Atom", "Molecule"]) -> "Molecule":
        if isinstance(other, Molecule):
            return Molecule(self, *other.atoms)
        elif isinstance(other, Atom):
            return Molecule(self, other)

    @classmethod
    def from_BSE(
        cls,
        basis:str,
        element:str,
        origin=np.zeros(3)
    ) -> "Atom":
        """ Initialize using values gathered from Basis Set Exchange.

            Args:
                basis (str): name of the basis to use
                element (str): name or principal quantum number of the element

            Returns:
                atom (Atom): resulting atom of given element in specified basis
        """
        import requests
        # call to api and parse response
        resp = requests.get("https://www.basissetexchange.org/api/basis/%s/format/json?elements=%s" % (basis, element))
        data = resp.json()
        # check
        assert all(ftype == 'gto' for ftype in data['function_types']), "Only GTO function type is supported!"
        assert len(data['elements']) == 1, "Expected only a single element but got %d!" % len(data['elements'])        
        # read element
        Z, data = next(iter(data['elements'].items()))
        alphas = [basis['exponents'] for basis in data['electron_shells']]
        coefficients = [basis['coefficients'] for basis in data['electron_shells']]
        angular_quantum_numbers = [basis['angular_momentum'] for basis in data['electron_shells']]
        # check values
        assert all(len(c) == len(a) for c, a in zip(coefficients, angular_quantum_numbers)), "Coefficients and angular momenta don't match up!"
        
        def yield_angular_momenta(aqn):
            # build all lists of three components with their sum 
            # matching the angular quantum number
            for i in range(aqn + 1):
                for j in range(aqn + 1 - i):
                    yield [i, j, aqn - i - j]

        # create atom
        return Atom(
            basis=[
                GaussianOrbital(
                    alpha=list(map(float, alpha)),
                    coeff=list(map(float, coeff)),
                    origin=origin,
                    angular=angular
                )
                for alpha, coeffs, aqns in zip(alphas, coefficients, angular_quantum_numbers)
                for aqn, coeff in zip(aqns, coeffs)
                for angular in yield_angular_momenta(aqn)
            ],
            Z=int(Z)
        )

class Molecule(object):
    """ Molecule described by a number of atoms 

        Args:
            *atoms (Atom): set of atoms building the molecule
    """

    def __init__(self, *atoms:Atom) -> None:
        # save atoms
        self.atoms = atoms

    @property
    def Zs(self) -> np.ndarray:
        """ Principal quantum numbers of all atoms. """
        return np.asarray([a.Z for a in self.atoms])

    @property
    def origins(self) -> np.ndarray:
        """ Origins of all atoms in shape (n, 3) where n is the number of atoms """
        return np.stack([a.origin for a in self.atoms], axis=0)
    
    @property
    def basis(self) -> List[GaussianOrbital]:
        """ Basis of the molecule, i.e. the union of all atom basis """
        return list(chain.from_iterable([a.basis for a in self.atoms]))

    def __len__(self) -> int:
        return len(self.atoms)

    def __add__(self, other:Union[Atom, "Molecule"]) -> "Molecule":
        if isinstance(other, Molecule):
            return Molecule(*self.atoms, *other.atoms)
        elif isinstance(other, Atom):
            return Molecule(*self.atoms, other)
