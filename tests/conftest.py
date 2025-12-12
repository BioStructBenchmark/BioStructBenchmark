"""
Shared test fixtures and helpers for GEMMI-compatible testing
"""

from unittest.mock import Mock


def create_mock_gemmi_residue(resname, seqid_num=1, icode=" ", chain_name="A"):
    """Create a GEMMI-compatible mock residue"""
    residue = Mock()
    residue.name = resname

    # Mock seqid (GEMMI uses seqid with num and icode attributes)
    seqid = Mock()
    seqid.num = seqid_num
    seqid.icode = icode if icode and icode.strip() else None
    residue.seqid = seqid

    # Mock het_flag for B-factor tests
    residue.het_flag = " "  # Standard residue

    # Mock atoms for the residue
    atoms = []
    atom = Mock()
    atom.name = "CA" if resname in ["ALA", "GLY", "VAL", "PHE", "TRP"] else "P"

    # GEMMI uses Position for atom.pos
    pos = Mock()
    pos.x = 0.0
    pos.y = 0.0
    pos.z = 0.0
    atom.pos = pos

    # GEMMI uses b_iso for B-factors
    atom.b_iso = 20.0

    atoms.append(atom)
    residue.__iter__ = lambda self: iter(atoms)

    return residue


def create_mock_gemmi_chain(chain_name, residues):
    """Create a GEMMI-compatible mock chain"""
    chain = Mock()
    chain.name = chain_name
    chain.__iter__ = lambda self: iter(residues)
    return chain


def create_mock_gemmi_model(chains):
    """Create a GEMMI-compatible mock model"""
    model = Mock()
    model.__iter__ = lambda self: iter(chains)
    return model


def create_mock_gemmi_structure(chains):
    """Create a GEMMI-compatible mock structure"""
    structure = Mock()
    model = create_mock_gemmi_model(chains)
    structure.__iter__ = lambda self: iter([model])
    structure.__len__ = lambda self: 1
    structure.__getitem__ = lambda self, idx: model if idx == 0 else None
    return structure
