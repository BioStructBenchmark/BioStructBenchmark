"""
biostructbenchmark/analysis/curves.py

CURVES+ integration for nucleic acid geometry analysis
Handles CURVES+ execution, output parsing, and geometry comparisons
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from Bio.PDB import Structure

from ..core.io import get_structure


@dataclass
class CurvesParameters:
    """Container for CURVES+ geometric parameters"""
    # Base pair parameters
    shear: float
    stretch: float
    stagger: float
    buckle: float
    propeller: float
    opening: float
    
    # Base pair step parameters
    shift: float
    slide: float
    rise: float
    tilt: float
    roll: float
    twist: float
    
    # Groove parameters
    major_groove_width: Optional[float] = None
    minor_groove_width: Optional[float] = None
    major_groove_depth: Optional[float] = None
    minor_groove_depth: Optional[float] = None


@dataclass
class HydrogenBond:
    """Hydrogen bond information"""
    donor_residue: str
    donor_atom: str
    acceptor_residue: str
    acceptor_atom: str
    distance: float
    angle: float
    strength: str  # 'strong', 'medium', 'weak'


class CurvesAnalyzer:
    """Main class for CURVES+ analysis and hydrogen bond detection"""
    
    def __init__(self, curves_executable: Optional[str] = None):
        """
        Initialize CURVES+ analyzer
        
        Args:
            curves_executable: Path to CURVES+ executable. If None, searches PATH
        """
        self.curves_exe = self._find_curves_executable(curves_executable)
        if not self.curves_exe:
            raise RuntimeError("CURVES+ executable not found. Please install CURVES+ and ensure it's in PATH")
    
    def _find_curves_executable(self, curves_path: Optional[str]) -> Optional[str]:
        """Find CURVES+ executable"""
        if curves_path and os.path.isfile(curves_path):
            return curves_path
        
        # Check common locations
        common_paths = [
            'Cur+',  # Standard name
            'curves+',
            '/usr/local/bin/Cur+',
            '/opt/curves/Cur+',
            os.path.expanduser('~/curves/Cur+')
        ]
        
        for path in common_paths:
            if shutil.which(path):
                return path
        
        return None
    
    def analyze_structure(self, structure_path: Path, 
                         output_dir: Optional[Path] = None) -> Dict[str, CurvesParameters]:
        """
        Run CURVES+ analysis on a structure
        
        Args:
            structure_path: Path to PDB/CIF file
            output_dir: Directory for output files (temporary if None)
            
        Returns:
            Dictionary mapping base pair steps to their parameters
        """
        # Create temporary directory if needed
        temp_dir = output_dir or Path(tempfile.mkdtemp(prefix='curves_'))
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract DNA chains to separate PDB file
            dna_pdb = self._extract_dna_chains(structure_path, temp_dir)
            
            # Run CURVES+
            curves_output = self._run_curves(dna_pdb, temp_dir)
            
            # Parse results
            parameters = self._parse_curves_output(curves_output)
            
            return parameters
            
        finally:
            # Clean up temporary files if we created them
            if output_dir is None:
                shutil.rmtree(temp_dir)
    
    def _extract_dna_chains(self, structure_path: Path, output_dir: Path) -> Path:
        """Extract DNA chains from structure file"""
        structure = get_structure(structure_path)
        if not structure:
            raise ValueError(f"Could not load structure from {structure_path}")
        
        # Create new structure with only DNA
        from Bio.PDB import PDBIO, Select
        
        class DNASelect(Select):
            def accept_residue(self, residue):
                # Accept DNA nucleotides
                return residue.get_resname().strip() in {
                    'DA', 'DT', 'DG', 'DC',  # DNA
                    'A', 'T', 'G', 'C',      # Alternative naming
                    'ADE', 'THY', 'GUA', 'CYT'  # Full names
                }
        
        dna_pdb = output_dir / f"{structure_path.stem}_dna.pdb"
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(dna_pdb), DNASelect())
        
        return dna_pdb
    
    def _run_curves(self, pdb_file: Path, output_dir: Path) -> Path:
        """Execute CURVES+ on DNA structure"""
        output_file = output_dir / f"{pdb_file.stem}_curves.out"
        
        # CURVES+ command with standard options
        cmd = [
            self.curves_exe,
            '-f', str(pdb_file),
            '-o', str(output_file),
            '-g',  # Generate all output files
            '-l',  # Line mode (non-interactive)
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=output_dir,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"CURVES+ failed: {result.stderr}")
            
            if not output_file.exists():
                raise RuntimeError("CURVES+ did not generate expected output file")
            
            return output_file
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("CURVES+ execution timed out")
        except FileNotFoundError:
            raise RuntimeError(f"CURVES+ executable not found: {self.curves_exe}")
    
    def _parse_curves_output(self, output_file: Path) -> Dict[str, CurvesParameters]:
        """Parse CURVES+ output files"""
        parameters = {}
        
        # CURVES+ generates multiple output files
        base_name = output_file.stem.replace('_curves', '')
        output_dir = output_file.parent
        
        # Parse base pair parameters (.bp file)
        bp_file = output_dir / f"{base_name}.bp"
        if bp_file.exists():
            bp_params = self._parse_bp_file(bp_file)
            parameters.update(bp_params)
        
        # Parse base pair step parameters (.bps file) 
        bps_file = output_dir / f"{base_name}.bps"
        if bps_file.exists():
            bps_params = self._parse_bps_file(bps_file)
            # Merge with bp_params
            for step, bps_data in bps_params.items():
                if step in parameters:
                    # Update existing CurvesParameters object
                    for attr, value in bps_data.items():
                        setattr(parameters[step], attr, value)
        
        # Parse groove parameters (.grv file)
        grv_file = output_dir / f"{base_name}.grv"
        if grv_file.exists():
            groove_params = self._parse_groove_file(grv_file)
            for step, groove_data in groove_params.items():
                if step in parameters:
                    for attr, value in groove_data.items():
                        setattr(parameters[step], attr, value)
        
        return parameters
    
    def _parse_bp_file(self, bp_file: Path) -> Dict[str, CurvesParameters]:
        """Parse base pair parameters file"""
        parameters = {}
        
        with open(bp_file) as f:
            lines = f.readlines()
        
        # Skip header lines
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#') or line.strip() == '':
                continue
            if 'Shear' in line and 'Stretch' in line:
                data_start = i + 1
                break
        
        # Parse data lines
        for line in lines[data_start:]:
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 7:
                step_id = f"{parts[0]}_{parts[1]}"  # e.g., "1_A-T"
                
                parameters[step_id] = CurvesParameters(
                    shear=float(parts[2]),
                    stretch=float(parts[3]),
                    stagger=float(parts[4]),
                    buckle=float(parts[5]),
                    propeller=float(parts[6]),
                    opening=float(parts[7]) if len(parts) > 7 else 0.0,
                    # Step parameters initialized to 0, will be updated from .bps file
                    shift=0.0, slide=0.0, rise=0.0,
                    tilt=0.0, roll=0.0, twist=0.0
                )
        
        return parameters
    
    def _parse_bps_file(self, bps_file: Path) -> Dict[str, Dict[str, float]]:
        """Parse base pair step parameters file"""
        step_params = {}
        
        with open(bps_file) as f:
            lines = f.readlines()
        
        # Find data section
        data_start = 0
        for i, line in enumerate(lines):
            if 'Shift' in line and 'Slide' in line and 'Rise' in line:
                data_start = i + 1
                break
        
        # Parse step parameters
        for line in lines[data_start:]:
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 7:
                step_id = f"{parts[0]}_{parts[1]}"  # e.g., "1_AT/TA"
                
                step_params[step_id] = {
                    'shift': float(parts[2]),
                    'slide': float(parts[3]),
                    'rise': float(parts[4]),
                    'tilt': float(parts[5]),
                    'roll': float(parts[6]),
                    'twist': float(parts[7]) if len(parts) > 7 else 0.0
                }
        
        return step_params
    
    def _parse_groove_file(self, grv_file: Path) -> Dict[str, Dict[str, float]]:
        """Parse groove parameters file"""
        groove_params = {}
        
        with open(grv_file) as f:
            lines = f.readlines()
        
        # Parse groove width and depth data
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                step_id = f"{parts[0]}_{parts[1]}"
                
                groove_params[step_id] = {
                    'major_groove_width': float(parts[2]),
                    'minor_groove_width': float(parts[3]),
                    'major_groove_depth': float(parts[4]) if len(parts) > 4 else None,
                    'minor_groove_depth': float(parts[5]) if len(parts) > 5 else None
                }
        
        return groove_params
    
    def detect_hydrogen_bonds(self, structure_path: Path, 
                            distance_cutoff: float = 3.5,
                            angle_cutoff: float = 120.0) -> List[HydrogenBond]:
        """
        Detect hydrogen bonds in DNA-protein interface
        
        Args:
            structure_path: Path to structure file
            distance_cutoff: Maximum H-bond distance (Angstroms)
            angle_cutoff: Minimum H-bond angle (degrees)
            
        Returns:
            List of detected hydrogen bonds
        """
        structure = get_structure(structure_path)
        if not structure:
            return []
        
        bonds = []
        
        # Get protein and DNA residues
        protein_residues = self._get_protein_residues(structure)
        dna_residues = self._get_dna_residues(structure)
        
        # Find potential H-bonds between protein and DNA
        for protein_res in protein_residues:
            for dna_res in dna_residues:
                res_bonds = self._find_residue_hbonds(
                    protein_res, dna_res, distance_cutoff, angle_cutoff
                )
                bonds.extend(res_bonds)
        
        return bonds
    
    def _get_protein_residues(self, structure: Structure.Structure) -> List:
        """Extract protein residues from structure"""
        protein_residues = []
        standard_aa = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
            'THR', 'TRP', 'TYR', 'VAL'
        }
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname().strip() in standard_aa:
                        protein_residues.append(residue)
        
        return protein_residues
    
    def _get_dna_residues(self, structure: Structure.Structure) -> List:
        """Extract DNA residues from structure"""
        dna_residues = []
        dna_bases = {
            'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C',
            'ADE', 'THY', 'GUA', 'CYT'
        }
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname().strip() in dna_bases:
                        dna_residues.append(residue)
        
        return dna_residues
    
    def _find_residue_hbonds(self, protein_res, dna_res, 
                           distance_cutoff: float, angle_cutoff: float) -> List[HydrogenBond]:
        """Find hydrogen bonds between two residues"""
        bonds = []
        
        # Define potential donors and acceptors
        # This is a simplified version - could be expanded with more sophisticated chemistry
        potential_donors = self._get_donors(protein_res) + self._get_donors(dna_res)
        potential_acceptors = self._get_acceptors(protein_res) + self._get_acceptors(dna_res)
        
        for donor_info in potential_donors:
            donor_atom, donor_res = donor_info
            
            for acceptor_info in potential_acceptors:
                acceptor_atom, acceptor_res = acceptor_info
                
                # Skip if same residue
                if donor_res == acceptor_res:
                    continue
                
                # Calculate distance
                try:
                    distance = donor_atom - acceptor_atom  # BioPython distance calculation
                    
                    if distance <= distance_cutoff:
                        # Calculate angle (simplified - assumes ideal geometry)
                        angle = self._calculate_hbond_angle(donor_atom, acceptor_atom)
                        
                        if angle >= angle_cutoff:
                            strength = self._classify_hbond_strength(distance, angle)
                            
                            bond = HydrogenBond(
                                donor_residue=f"{donor_res.get_resname()}_{donor_res.get_id()[1]}",
                                donor_atom=donor_atom.get_name(),
                                acceptor_residue=f"{acceptor_res.get_resname()}_{acceptor_res.get_id()[1]}",
                                acceptor_atom=acceptor_atom.get_name(),
                                distance=distance,
                                angle=angle,
                                strength=strength
                            )
                            bonds.append(bond)
                
                except Exception:
                    # Skip problematic atom pairs
                    continue
        
        return bonds
    
    def _get_donors(self, residue) -> List[Tuple]:
        """Get potential hydrogen bond donors from residue"""
        donors = []
        
        # Protein donors
        if residue.get_resname() in ['ARG', 'LYS', 'HIS', 'ASN', 'GLN', 'SER', 'THR', 'TYR', 'TRP']:
            for atom in residue:
                if atom.get_name().startswith('N') or atom.get_name().startswith('O'):
                    if atom.get_name() in ['NH1', 'NH2', 'NZ', 'ND1', 'NE2', 'ND2', 'NE1', 'OG', 'OG1', 'OH']:
                        donors.append((atom, residue))
        
        # DNA donors (simplified)
        if residue.get_resname().strip() in ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']:
            for atom in residue:
                if atom.get_name() in ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9']:
                    donors.append((atom, residue))
        
        return donors
    
    def _get_acceptors(self, residue) -> List[Tuple]:
        """Get potential hydrogen bond acceptors from residue"""
        acceptors = []
        
        # Protein acceptors
        for atom in residue:
            if atom.get_name().startswith('O') or atom.get_name().startswith('N'):
                if atom.get_name() in ['O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'ND1', 'NE2']:
                    acceptors.append((atom, residue))
        
        # DNA acceptors
        if residue.get_resname().strip() in ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']:
            for atom in residue:
                if atom.get_name() in ['O2', 'O4', 'O6', 'N1', 'N3', 'N7']:
                    acceptors.append((atom, residue))
        
        return acceptors
    
    def _calculate_hbond_angle(self, donor_atom, acceptor_atom) -> float:
        """Calculate hydrogen bond angle (simplified)"""
        # This is a simplified calculation
        # In reality, you'd want to consider the hydrogen position and geometry
        return 150.0  # Placeholder - would need proper geometric calculation
    
    def _classify_hbond_strength(self, distance: float, angle: float) -> str:
        """Classify hydrogen bond strength"""
        if distance < 2.8 and angle > 160:
            return 'strong'
        elif distance < 3.2 and angle > 140:
            return 'medium'
        else:
            return 'weak'
    
    def compare_geometries(self, experimental_params: Dict[str, CurvesParameters],
                         predicted_params: Dict[str, CurvesParameters]) -> pd.DataFrame:
        """
        Compare geometric parameters between experimental and predicted structures
        
        Returns:
            DataFrame with parameter differences
        """
        comparisons = []
        
        # Find common base pairs/steps
        common_steps = set(experimental_params.keys()) & set(predicted_params.keys())
        
        for step in common_steps:
            exp_params = experimental_params[step]
            pred_params = predicted_params[step]
            
            # Calculate differences for each parameter
            comparison = {
                'step': step,
                'shear_diff': pred_params.shear - exp_params.shear,
                'stretch_diff': pred_params.stretch - exp_params.stretch,
                'stagger_diff': pred_params.stagger - exp_params.stagger,
                'buckle_diff': pred_params.buckle - exp_params.buckle,
                'propeller_diff': pred_params.propeller - exp_params.propeller,
                'opening_diff': pred_params.opening - exp_params.opening,
                'shift_diff': pred_params.shift - exp_params.shift,
                'slide_diff': pred_params.slide - exp_params.slide,
                'rise_diff': pred_params.rise - exp_params.rise,
                'tilt_diff': pred_params.tilt - exp_params.tilt,
                'roll_diff': pred_params.roll - exp_params.roll,
                'twist_diff': pred_params.twist - exp_params.twist,
            }
            
            # Add groove comparisons if available
            if (exp_params.major_groove_width is not None and 
                pred_params.major_groove_width is not None):
                comparison['major_groove_width_diff'] = (
                    pred_params.major_groove_width - exp_params.major_groove_width
                )
                comparison['minor_groove_width_diff'] = (
                    pred_params.minor_groove_width - exp_params.minor_groove_width
                )
            
            comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)


def analyze_dna_geometry_batch(experimental_dir: Path, predicted_dir: Path,
                              output_dir: Path) -> None:
    """
    Batch analysis of DNA geometry for structure pairs
    
    Args:
        experimental_dir: Directory containing experimental structures
        predicted_dir: Directory containing predicted structures  
        output_dir: Directory for results
    """
    analyzer = CurvesAnalyzer()
    output_dir.mkdir(exist_ok=True)
    
    # Find matching structure pairs
    exp_files = list(experimental_dir.glob("*.pdb")) + list(experimental_dir.glob("*.cif"))
    
    all_comparisons = []
    
    for exp_file in exp_files:
        # Find corresponding predicted structure
        pred_file = None
        for ext in ['.pdb', '.cif']:
            potential_pred = predicted_dir / f"{exp_file.stem}{ext}"
            if potential_pred.exists():
                pred_file = potential_pred
                break
        
        if not pred_file:
            print(f"Warning: No predicted structure found for {exp_file.name}")
            continue
        
        print(f"Analyzing: {exp_file.name} vs {pred_file.name}")
        
        try:
            # Analyze both structures
            exp_params = analyzer.analyze_structure(exp_file)
            pred_params = analyzer.analyze_structure(pred_file)
            
            # Compare parameters
            comparison_df = analyzer.compare_geometries(exp_params, pred_params)
            comparison_df['structure_pair'] = exp_file.stem
            
            all_comparisons.append(comparison_df)
            
            # Analyze hydrogen bonds
            exp_hbonds = analyzer.detect_hydrogen_bonds(exp_file)
            pred_hbonds = analyzer.detect_hydrogen_bonds(pred_file)
            
            # Save individual results
            structure_output = output_dir / exp_file.stem
            structure_output.mkdir(exist_ok=True)
            
            comparison_df.to_csv(structure_output / 'geometry_comparison.csv', index=False)
            
            # Save hydrogen bond data
            exp_hbonds_df = pd.DataFrame([vars(hb) for hb in exp_hbonds])
            pred_hbonds_df = pd.DataFrame([vars(hb) for hb in pred_hbonds])
            
            exp_hbonds_df.to_csv(structure_output / 'experimental_hbonds.csv', index=False)
            pred_hbonds_df.to_csv(structure_output / 'predicted_hbonds.csv', index=False)
            
        except Exception as e:
            print(f"Error processing {exp_file.name}: {e}")
            continue
    
    # Combine all comparisons
    if all_comparisons:
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        combined_df.to_csv(output_dir / 'all_geometry_comparisons.csv', index=False)
        
        # Generate summary statistics
        summary_stats = combined_df.select_dtypes(include=[np.number]).describe()
        summary_stats.to_csv(output_dir / 'geometry_summary_statistics.csv')
        
        print(f"Analysis complete. Results saved to {output_dir}")
    else:
        print("No successful analyses completed")


if __name__ == "__main__":
    # Example usage
    analyzer = CurvesAnalyzer()
    
    # Test on a single structure
    test_structure = Path("test_structure.pdb")
    if test_structure.exists():
        params = analyzer.analyze_structure(test_structure)
        hbonds = analyzer.detect_hydrogen_bonds(test_structure)
        
        print(f"Found {len(params)} base pair steps")
        print(f"Found {len(hbonds)} hydrogen bonds")
