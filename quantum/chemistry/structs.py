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
        assert all(o.origin == basis[0].origin for o in basis[1:]), "Origins don't match!"
    
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
