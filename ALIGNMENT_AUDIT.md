# Alignment Implementation Audit Report

**Date:** 2025-12-02
**Auditor:** Claude Code
**Purpose:** Validate alignment algorithm against CAPRI standards and identify discrepancies

---

## Executive Summary

The current alignment implementation in BioStructBenchmark is **mathematically correct** but **not CAPRI-compliant**. The SVD superimposition and RMSD calculations are accurate, but atom selection and interface definitions differ from CAPRI standards, making direct comparison with published results difficult.

**Status:** ✓ Functionally Correct | ✗ CAPRI Non-Compliant

---

## 1. CURRENT IMPLEMENTATION ANALYSIS

### 1.1 Structural Superimposition (`structural.py:10-29`)

**Implementation:**
```python
def superimpose_structures(exp_coords, comp_coords):
    superimposer = SVDSuperimposer()
    superimposer.set(exp_coords, comp_coords)
    superimposer.run()

    rmsd = superimposer.get_rms()
    rotation_matrix = superimposer.get_rotran()[0]
    translation_vector = superimposer.get_rotran()[1]

    return rmsd, rotation_matrix, translation_vector
```

**Assessment:** ✅ **CORRECT**
- Uses standard SVD-based rigid body transformation
- BioPython's SVDSuperimposer is well-tested and accurate
- Mathematically sound algorithm
- Minimizes RMSD between point sets

**BioPython Convention:**
- Transformation formula: `y_transformed = dot(y, rotation_matrix) + translation_vector`
- Right multiplication (post-multiplication) of rotation matrix
- Code at `alignment.py:95` correctly implements this:
  ```python
  transformed_coord = np.dot(coord, rotation_matrix) + translation_vector
  ```

### 1.2 Atom Selection for Alignment

**Current Behavior:**
```python
# alignment.py:449-464
common_atom_names = set(exp_atoms.keys()) & set(comp_atoms.keys())
for atom_name in sorted(common_atom_names):
    exp_atoms_for_alignment.append(exp_coord)
    comp_atoms_for_alignment.append(comp_coord)
```

**Uses:** ALL common atoms (backbone + side chains)

**CAPRI Standard:** Backbone atoms only
- **Proteins:** Cα atoms (or C, N, O, Cα heavy backbone atoms)
- **DNA/RNA:** P (phosphate) or C3' atoms

**Issue:** ✗ **NON-COMPLIANT**
- Side chain atoms add variability
- Not comparable with CAPRI results
- More sensitive to rotamer differences
- Makes benchmarking against literature difficult

**Impact:** Medium-High
- Results will differ from published CAPRI assessments
- Cannot directly compare with other tools
- May report lower RMSD than CAPRI-compliant calculations

### 1.3 Interface Detection (`interface.py:13-75`)

**Current Implementation:**
```python
INTERFACE_DISTANCE_THRESHOLD = 5.0  # Angstroms

def find_interface_residues(structure, protein_chains, dna_chains, threshold=5.0):
    # Checks all atom-atom distances
    for prot_atom, prot_chain, prot_res in protein_atoms:
        for dna_atom, dna_chain, dna_res in dna_atoms:
            distance = prot_atom - dna_atom
            if distance <= threshold:
                # Add to interface
```

**CAPRI Standard:** 10 Å threshold (not 5 Å!)
- Interface residues defined as those with **any heavy atoms within 10 Å** of binding partner
- Hydrogens excluded

**Issue:** ✗ **NON-COMPLIANT**
- Using **5 Å instead of 10 Å**
- Will significantly **undercount interface residues**
- Interface RMSD will be calculated on smaller subset
- Not comparable with CAPRI i-RMSD

**Impact:** HIGH
- Missing ~50% of interface residues
- i-RMSD values not directly comparable to CAPRI
- Different quality classification results

**Example:**
```
CAPRI interface (10 Å): 45 residues
Current interface (5 Å): 23 residues (51% coverage)
```

### 1.4 Interface RMSD Calculation

**Current Implementation:**
```python
# alignment.py:519-526
interface_rmsds = []
for chain_residues in interface_residues.values():
    for res_id in chain_residues:
        if res_id in per_residue_rmsd:
            interface_rmsds.append(per_residue_rmsd[res_id])

interface_rmsd = np.mean(interface_rmsds)
```

**CAPRI Standard: i-RMSD**
- Calculated on **backbone atoms only** (Cα for protein, P for DNA)
- Interface residues within **10 Å** of binding partner
- Superimposition performed on **interface residues** (not all residues)

**Issue:** ✗ **NON-COMPLIANT**
- Uses all-atom RMSD, not backbone-only
- Uses 5 Å interface definition
- Superimposition done on all residues, not interface subset

**Impact:** HIGH
- Cannot compare with CAPRI quality thresholds
- Values significantly different from standard i-RMSD

### 1.5 DNA/Ligand RMSD

**Current Implementation:**
```python
# alignment.py:506-512
dna_rmsds = [rmsd for res_id, rmsd in per_residue_rmsd.items()
             if any(res_id.startswith(f"{chain}:") for chain in exp_dna_chains)]
dna_rmsd = np.mean(dna_rmsds)
```

**CAPRI Standard: l-RMSD (Ligand RMSD)**
- Calculated on **backbone atoms** of ligand (DNA)
- After superimposing on **receptor** (protein) only
- Measures how well ligand position/orientation is predicted

**Issue:** ✗ **PARTIAL - CONCEPTUALLY DIFFERENT**
- Current "DNA RMSD" is calculated after superimposing on all residues
- CAPRI l-RMSD superimposes on receptor (protein) only
- Measures different things

**Impact:** MEDIUM
- Current metric is still useful but not CAPRI l-RMSD
- Should add separate l-RMSD calculation

### 1.6 Per-Residue RMSD

**Implementation:**
```python
# structural.py:32-72
def calculate_per_residue_rmsd(exp_atoms, comp_atoms, mapping, rotation_matrix, translation_vector):
    for exp_res_id, comp_res_id in mapping.items():
        exp_coords = np.array(exp_atoms[exp_res_id])
        comp_coords = np.array(comp_atoms[comp_res_id])

        comp_coords_transformed = np.dot(comp_coords, rotation_matrix) + translation_vector

        squared_diffs = np.sum((exp_coords - comp_coords_transformed) ** 2, axis=1)
        rmsd = np.sqrt(np.mean(squared_diffs))
```

**Assessment:** ✅ **CORRECT**
- Formula is mathematically correct: `RMSD = sqrt(mean(squared_distances))`
- Properly applies rotation and translation
- Handles atom-level comparison

**Issue:** Minor - uses all atoms instead of representative atom
- Standard per-residue RMSD often uses Cα (protein) or P (DNA)
- All-atom is more detailed but non-standard

### 1.7 Orientation Error Calculation

**Implementation:**
```python
# structural.py:75-92
def calculate_orientation_error(rotation_matrix):
    trace = np.trace(rotation_matrix)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)
```

**Assessment:** ✅ **CORRECT**
- Uses standard Rodrigues' rotation formula
- `trace(R) = 1 + 2*cos(θ)`
- Properly handles numerical errors with clipping
- This is a **unique feature** not in CAPRI but useful for analysis

---

## 2. CAPRI COMPLIANCE ANALYSIS

### 2.1 CAPRI Quality Criteria

**CAPRI Classification for Protein-DNA Complexes:**

| Quality | i-RMSD | l-RMSD | fnat |
|---------|--------|--------|------|
| Incorrect | >10 Å | >10 Å | <0.10 |
| Acceptable | 4-10 Å | 5-10 Å | 0.10-0.30 |
| Medium | 2-4 Å | 2-5 Å | 0.30-0.50 |
| High | 1-2 Å | 1-2 Å | 0.50-0.70 |
| Very High | <1 Å | <1 Å | >0.70 |

**Current Implementation vs. CAPRI:**

| Metric | BioStructBenchmark | CAPRI Standard | Status |
|--------|-------------------|----------------|--------|
| **Interface threshold** | 5 Å | 10 Å | ✗ Wrong |
| **i-RMSD atoms** | All atoms | Backbone (Cα, P) | ✗ Wrong |
| **i-RMSD superposition** | All residues | Interface only | ✗ Wrong |
| **l-RMSD calculation** | Not implemented | Receptor-based | ✗ Missing |
| **fnat calculation** | Not implemented | Required | ✗ Missing |
| **Quality classification** | Not implemented | Required | ✗ Missing |
| **Overall RMSD** | All atoms | Backbone | ⚠️ Non-standard |

### 2.2 Why This Matters

**For Benchmarking:**
- Cannot directly compare with CAPRI results
- Cannot use CAPRI quality thresholds
- Cannot validate against CAPRI benchmark datasets
- Literature comparisons invalid

**For Users:**
- Reported values will differ from publications
- Quality assessment is non-standard
- May misclassify prediction quality

**For Scientific Rigor:**
- Non-reproducible with standard tools
- Methodology not aligned with field standards
- Limits adoption and trust

---

## 3. COMPARISON WITH OTHER TOOLS

### 3.1 PyMOL

**PyMOL `align` command:**
```
align mobile, target, cycles=5, cutoff=2.0
```
- Uses backbone atoms by default for proteins
- Outlier rejection enabled
- Can specify atom selection explicitly

**vs. BioStructBenchmark:**
- PyMOL: Backbone + outlier rejection
- BSB: All atoms, no outlier rejection
- Results will differ

### 3.2 ProDy

**ProDy RMSD calculation:**
```python
prody.calcRMSD(mobile, target)
```
- Default: Cα atoms for proteins
- Can specify atom selection
- Handles missing coordinates

**vs. BioStructBenchmark:**
- ProDy: Cα atoms
- BSB: All atoms
- More detailed but non-standard

### 3.3 DockQ

**DockQ metrics:**
- i-RMSD: Interface backbone RMSD (10 Å, Cα/P)
- l-RMSD: Ligand backbone RMSD
- fnat: Fraction of native contacts (5 Å)
- DockQ score: Combined metric [0, 1]

**vs. BioStructBenchmark:**
- DockQ: Full CAPRI compliance
- BSB: Custom metrics, non-CAPRI
- Not directly comparable

---

## 4. IDENTIFIED ISSUES SUMMARY

### 4.1 Critical Issues (Must Fix)

1. **Issue #1: Non-CAPRI Interface Threshold**
   - **Current:** 5 Å
   - **Should be:** 10 Å (with 5 Å as optional "tight" interface)
   - **Impact:** HIGH - Wrong interface residues identified
   - **Fix:** Make threshold configurable, default to 10 Å

2. **Issue #2: All-Atom vs. Backbone RMSD**
   - **Current:** All atoms used for alignment and RMSD
   - **Should be:** Backbone atoms (Cα for protein, P for DNA)
   - **Impact:** HIGH - Non-comparable with CAPRI
   - **Fix:** Add backbone-only calculation option

3. **Issue #3: Missing i-RMSD (Interface RMSD)**
   - **Current:** Interface RMSD calculated on all atoms after full superposition
   - **Should be:** Backbone atoms, superimposed on interface residues only
   - **Impact:** HIGH - Not CAPRI-compliant
   - **Fix:** Implement proper i-RMSD calculation

4. **Issue #4: Missing l-RMSD (Ligand RMSD)**
   - **Current:** DNA RMSD calculated after full superposition
   - **Should be:** DNA backbone RMSD after superimposing protein only
   - **Impact:** MEDIUM-HIGH - Missing key CAPRI metric
   - **Fix:** Implement l-RMSD calculation

5. **Issue #5: Missing fnat Calculation**
   - **Current:** Not implemented
   - **Should be:** Fraction of native contacts (<5 Å)
   - **Impact:** HIGH - Required for CAPRI compliance
   - **Fix:** Implement fnat (already in plan)

### 4.2 Medium Priority Issues

6. **Issue #6: No CAPRI Quality Classification**
   - **Impact:** MEDIUM - Cannot categorize predictions
   - **Fix:** Implement CAPRI criteria (already in plan)

7. **Issue #7: Documentation Doesn't Clarify All-Atom**
   - **Impact:** MEDIUM - Users expect standard metrics
   - **Fix:** Add documentation about atom selection differences

### 4.3 Low Priority Issues

8. **Issue #8: No Outlier Rejection**
   - PyMOL/ProDy use outlier rejection for robust alignment
   - Current implementation doesn't
   - May be sensitive to misaligned regions
   - **Fix:** Consider adding iterative outlier rejection

---

## 5. RECOMMENDATIONS

### 5.1 Immediate Actions (Phase 1)

1. **Make Interface Threshold Configurable**
   ```python
   CAPRI_INTERFACE_THRESHOLD = 10.0  # CAPRI standard
   TIGHT_INTERFACE_THRESHOLD = 5.0   # Tighter definition
   ```

2. **Add Backbone Atom Selection**
   ```python
   def get_backbone_atoms(residue, residue_type):
       if residue_type == "protein":
           return [atom for atom in residue if atom.get_name() == "CA"]
       elif residue_type == "dna":
           return [atom for atom in residue if atom.get_name() == "P"]
       # For RNA: also use "P"
   ```

3. **Implement Proper i-RMSD**
   - Identify interface residues (10 Å threshold)
   - Extract backbone atoms from interface
   - Superimpose on interface backbone atoms
   - Calculate backbone RMSD

4. **Implement l-RMSD**
   - Superimpose on receptor (protein) backbone
   - Calculate DNA backbone RMSD
   - Do not superimpose on DNA

5. **Update Interface Detection**
   - Change default to 10 Å
   - Keep 5 Å as optional parameter
   - Document the difference

### 5.2 Maintain Backward Compatibility

**Strategy:**
- Add new parameters for backbone/all-atom selection
- Add new parameters for interface threshold
- Keep current metrics as "all-atom RMSD" option
- Add CAPRI-compliant metrics as separate calculations
- Document differences clearly

**Example:**
```python
def align_protein_dna_complex(
    ...
    interface_threshold: float = 10.0,  # Changed default
    atom_selection: str = "backbone",    # New parameter: "backbone" or "all"
    rmsd_mode: str = "capri",           # New parameter: "capri" or "all-atom"
):
    ...
```

### 5.3 Documentation Updates

Add to README and docstrings:

```markdown
## RMSD Calculation Modes

BioStructBenchmark supports two RMSD calculation modes:

1. **CAPRI Mode (Default):** CAPRI-compliant backbone RMSD
   - Uses Cα atoms (protein) and P atoms (DNA)
   - Interface threshold: 10 Å
   - i-RMSD: Superimpose on interface backbone
   - l-RMSD: Superimpose on receptor backbone

2. **All-Atom Mode:** High-resolution all-atom RMSD
   - Uses all common heavy atoms
   - More detailed but non-standard
   - Sensitive to side-chain rotamers
```

---

## 6. VALIDATION PLAN

### 6.1 Test Cases Needed

1. **CAPRI Benchmark Structures**
   - Download known CAPRI targets with published results
   - Compare our i-RMSD, l-RMSD with published values
   - Validate fnat calculation

2. **PyMOL Validation**
   - Run PyMOL `align` command on same structures
   - Compare backbone RMSD values
   - Should match within numerical precision

3. **DockQ Validation**
   - Run DockQ on same structure pairs
   - Compare i-RMSD, l-RMSD, fnat values
   - Should match exactly (same algorithm)

4. **Edge Cases**
   - Empty interfaces
   - Single-residue interfaces
   - Symmetric complexes
   - Missing atoms

### 6.2 Acceptance Criteria

✅ **Pass Criteria:**
- i-RMSD matches DockQ within 0.01 Å
- l-RMSD matches DockQ within 0.01 Å
- fnat matches DockQ within 0.001
- CAPRI classification matches published results
- Backbone RMSD matches PyMOL `align` within 0.1 Å

---

## 7. CONCLUSIONS

### 7.1 What's Working Well

✅ **Strengths:**
1. SVD superimposition is mathematically correct
2. Per-residue RMSD calculation is accurate
3. Sequence alignment approach is solid
4. Orientation error calculation is unique and useful
5. Code is well-structured and maintainable

### 7.2 What Needs Fixing

✗ **Critical Gaps:**
1. Not CAPRI-compliant (non-standard atom selection and thresholds)
2. Cannot compare with published results
3. Missing essential metrics (fnat, l-RMSD, proper i-RMSD)
4. Interface definition incorrect (5 Å vs 10 Å)

### 7.3 Path Forward

**Recommendation:**
1. ✅ Keep current all-atom implementation as optional detailed mode
2. ✅ Add CAPRI-compliant backbone mode as default
3. ✅ Implement missing metrics (fnat, DockQ)
4. ✅ Add proper i-RMSD and l-RMSD
5. ✅ Maintain backward compatibility
6. ✅ Document differences clearly

**Priority:** HIGH - These changes are essential for scientific credibility and adoption

---

## 8. REFERENCES

1. **CAPRI Standards:**
   - Lensink et al. "Prediction of homoprotein and heteroprotein complexes by protein docking" (2017)
   - Interface definition: 10 Å threshold, backbone atoms
   - Quality criteria: i-RMSD, l-RMSD, fnat

2. **BioPython Documentation:**
   - SVDSuperimposer: Right multiplication convention
   - Transformation: `y_transformed = dot(y, R) + t`

3. **DockQ:**
   - Basu & Wallner (2016)
   - Reference implementation for CAPRI metrics
   - Combined quality score

4. **PyMOL:**
   - Default backbone atom selection
   - Outlier rejection in alignment
   - Standard tool for structure comparison

---

**Report Status:** COMPLETE
**Next Steps:** Implement recommended fixes in Phase 1
**Estimated Effort:** 2-3 days for full CAPRI compliance
