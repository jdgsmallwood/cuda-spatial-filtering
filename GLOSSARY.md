# Adaptive beamforming

Spatial RFI suppression for beamformed observations, including the correlation
needed to estimate spatial filters.

## Language

**Coarse channel**:
A frequency channel supplied by the FPGA channelizer, before further fine
channelization.

**Fine channel**:
A narrower frequency channel formed by subdividing a coarse channel.

**Decomposition granularity**:
The frequency resolution at which separate covariance eigendecompositions are
computed: coarse-channel or fine-channel resolution.

**Nulling rank**:
The number of spatial eigenmodes removed by a filter.
_Avoid_: Number of ranks.

**Eigenmode history**:
A time series of eigenvalues and their associated eigenvectors, used to study
which spatial modes to suppress or preserve.

**Protected solar signal**:
The desired astronomical contribution from the Sun that RFI mitigation must
preserve.

**Rank-1 source protection**:
Preservation of an assumed unresolved source's spatial signature by selecting
the nulled modes from the covariance restricted to that signature's orthogonal
complement. Rank one describes the protected signature, not the nulling rank.
