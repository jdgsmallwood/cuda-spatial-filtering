# Starspin JOSS readiness

Prepared 15 October 2026 against the current [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html),
[paper format](https://joss.readthedocs.io/en/latest/paper.html), and
[review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).
The submission is **not yet cleared for submission**. This checklist distinguishes
prepared materials from verification and author decisions still needed.

## Scope and research use

The submission covers the Starspin correlator: legacy/CODIF packet ingest,
reassembly and delay alignment, stream mapping, optional fine channelization,
TCC correlation, integration, and HDF5 provenance. Companion beamforming,
RFI mitigation, pulsar tools, and experimental ingest modes are not claimed as
validated Starspin features merely because they share this repository.

The maintainer confirmed current use for LAMBDA telescope commissioning on
15 October 2026. Site-specific commissioning records and calibration files are
maintained outside the public repository. Research use is maintainer-reported;
no public live-performance claim is inferred from the private records.

Authors supplied by the maintainer: Justin Smallwood, Adam Deller, Andrew Jameson,
and Chris Phillips. Affiliations, Justin Smallwood's ORCID, pipeline AI model
history, and acknowledgements were taken at the maintainer's direction from
`phd-thesis-writing/Paper1/Smallwood_Protection_Astronomical_Sources.tex`
(author block and Acknowledgements). Funding is CSIRO's Industry PhD program
in partnership with Fourier Space. No grant identifiers or other ORCIDs were
invented. That related manuscript reports earlier pipeline review; it does not
constitute sign-off on these new preparation changes.

## Evidence and remaining gates

| Item | Status |
| --- | --- |
| MIT licence and contribution/support instructions | Present |
| Public source and readable issue tracker | GitHub API reported public, issues enabled, 15 October 2026 |
| Iterative source history | Local history spans May 2025 to October 2026, distributed across months |
| More than six months of public development | Maintainer confirmed on 27 October 2026 that the repository has been public throughout its development history |
| Research use | LAMBDA commissioning confirmed; author-approved public evidence package still needed |
| Author list | Supplied with affiliations and one ORCID from the related paper; contributions and final consent still to confirm |
| Software citation | Valid CFF 1.2.0; no fabricated article DOI or release date |
| Clean recursive clone | Blocked by unavailable TCC gitlink; maintainer authorized local revision for preparation |
| Core Docker environment | Built from tracked Dockerfile and requirements; fresh source build and replay exercised with local TCC exception |
| Python checks | 27 October candidate: 49 passed, 4 optional skips on Python 3.10 with DiFX; fresh Python 3.12 environment: 48 passed, 5 skips (DiFX not enabled there) |
| GPU numerical replay | Freshly compiled binary passed independent reference; details below |
| CI | Local Python 3.12 tests and citation validation passed on 27 October; GitHub reports zero runs for the candidate branch; hosted execution requires committing/pushing the candidate and workflows |
| Paper | Draft includes required sections and references; visible author-confirmation markers remain |
| Independent colleague rehearsal | Pending; preparation runs by the maintainer's assistant do not satisfy this |
| Release and DOI | Public remote has no tags or GitHub releases as of 15 October; prepare after gates pass |

### Dependency exception

The superproject records TCC `d71f1b4d0f1dc03f02bedde738fc9b80818a42fd`.
On 15 October, recursive initialization failed because ASTRON's remote did not
contain that object. Local source at `36e5edc3bc49772670edb5451f7b397342d26602`
was restored at the maintainer's direction for continued verification. The
superproject gitlink has not been staged to a replacement revision. Publish the
intended TCC source through an accessible remote, record the correct revision,
and repeat a fresh recursive clone and full verification before submission.
Do not silently turn this local exception into a release dependency.

### Historical validation record (before synthetic-fixture replacement)

- Baseline superproject: `b12d9ec` plus the submission-preparation changes.
- Python: isolated environment, tracked Python tests only; 36 passed, 4 skipped.
  Three skips require optional `lambda_commissioning`; one requires the optional
  CUDA delay-test library. Unrelated untracked CryoPAF work was not included.
- Fresh C++/CUDA suites: **73 passed**: Starspin 6, fine-channelized Starspin 11,
  packet/stream-map/CODIF parsing 28, packet reassembly/CODIF ingest 21, writers 6,
  and run audit 1. These are the core reviewer suites, not every companion or
  hardware-specific test in the repository.
- Paper: official Inara renderer compiled a three-page draft; all pages visually
  inspected. Author-confirmation markers are intentional pending inputs.
- CFF: `cffconvert --validate` passed.
- Fresh `build_joss_clean/apps/starspin_1_8` replay: 30,650,400 complex visibilities,
  maximum relative RMS `8.468629503711667e-6`, maximum autopower-normalized error
  `2.1417335667641704e-5`; no missing packets in compared integrations.
  Conjugation and missing-scale negative controls exceeded tolerance.
  Evidence: `build_joss_clean/reviewer-replay/reference/comparison.json`.
- Existing binary replay also passed (31,648,320 complex visibilities); it was
  used first to verify the runner while the fresh environment was being built.
- Fixture SHA-256: `1aa9278e096a44c85be7c9a017f57495e449ca66c22fcdaf1f600277403f8a09`.
- Existing binary SHA-256: `ad118f705b8052cdd2f9739da68ad97fa6790318d07fc888deb0a6893a47dc94`.
- Existing binary evidence: `build/joss-existing-replay/`; this verifies the
  reviewer procedure, not a clean rebuild or current live throughput.
- Dependency downloads used Docker host networking on this machine; numerical
  replay and test execution used networking disabled.
- Fresh environment: Ubuntu 22.04, CUDA 12.8.93, GCC 11, CMake 3.31.6,
  RTX 4060 (sm_89, 8 GiB), NVIDIA driver 580.173.02. The Python tests also passed
  inside the new image with the same four explicit skips.
- Container image ID: `sha256:350c5c558a85a8f0218bdf3e241cc82a1b61d6282db408cfddd0e0f7bdf0f9ff`.
- Preparation logs and package inventory: `build/joss-evidence/` (local artifacts).
- No new telescope capture or FPGA changes were performed.

## External correlation check (19 October 2026)

A freshly built coarse-only Starspin target passed comparison with DiFX 2.9.0's
`zerocorr`: 11 complete integrations, 19,360 complex visibilities, maximum
relative RMS `2.3708038444005157e-8`, and maximum autopower-normalized error
`9.246423825285717e-8`. All compared integrations had zero missing packets;
conjugation, missing-scale, and polarization-swap controls were rejected.
The 15 targeted Python tests passed, including the actual DiFX compatibility
test. See [DiFX verification](difx-validation.md) for commands, provenance,
scope, and remaining coverage. This is synthetic coarse-only validation with
DiFX's utility, not `mpifxcorr`, live delay calibration, or fine-PFB parity.
It retains the existing local TCC dependency exception and does not close the
clean-clone or independent-colleague gates.

## Author completion

Replace every `AUTHOR CONFIRMATION REQUIRED` marker in `paper.md` with approved
information. Confirm author order, contributions and consent; add other ORCIDs if provided.
Affiliations and funding have been copied from the related paper. Confirm sponsor
involvement, conflicts, and the final preparation AI model identifier. Authors
must actually review and validate the work before asserting human approval.
Keep the commissioning impact claim specific to actual use. Do not claim
external observatory adoption, publications, or independent reproduction without
evidence. The public CODIF fixture is generated by `scripts/generate_codif_fixture.py`;
retain that generator and its reproducibility test in the archive.

The paper format is 750–1,750 words. Build it with the Paper workflow or the
official Open Journals Inara container, inspect the PDF, and resolve all citation
warnings. The author-facing draft is deliberately marked pending where details
are unknown; a compiling PDF alone is not submission readiness.

## Release and archive procedure

1. Resolve the public TCC dependency and repeat the reviewer guide from an
   independent recursive clone. Preserve exact source/submodule revisions,
   container image digest, GPU/driver, test logs, and replay comparison JSON.
2. Complete the colleague rehearsal and author review. Update this checklist,
   the paper, changelog, and citation metadata to match the verified candidate.
3. Check licences and notices for all redistributed dependencies and fixture
   permissions. GitHub source archives do not contain submodule contents;
   preserve dependency source/revisions or separately archived dependencies.
4. Create a release candidate using the agreed version and exact verified
   commit. Confirm public source, issues, contribution instructions, and CI.
5. The corresponding author completes the JOSS submission form and manages
   editor/reviewer correspondence. Current JOSS policy excludes AI-generated
   conversational interactions with editors/reviewers except translation.
6. After review changes, create the final tagged release and archive it with
   Zenodo or an equivalent service. Check archive completeness, obtain its
   version-specific DOI, and record that DOI and version in the review.
7. Add the accepted paper citation/DOI only after assignment; preserve the
   distinction between software-version DOI and paper DOI.

No submission, release publication, tag, DOI, or author approval has been
represented as completed by this preparation work.

## Review follow-up (15 October 2026)

The cleanup removed obsolete commented-out packet processing and unused CMake
scaffolding, shortened the gpu-filter workaround explanation, and corrected
receiver-padding, shared-filter, and test-run instructions. Paper claims remain
limited to the tested Starspin path; author confirmation is still pending.

Review findings and remaining coverage gaps:

- Companion arrival-mask ownership was corrected after this review: metadata is
  copied before input release, with deterministic regression tests in
  `PipelineHarnessSelfTest` and `AdaptivePipelineTests` (see `tests/TESTING.md`).
- The three-second offline replay does not establish sustained buffer-reuse
  behavior under live NIC load, GPUDirect correctness, or production throughput.
- Optional commissioning-package integrations and the GPU delay-library Python
  check remain skipped in the core environment. Run them in their supported
  environment before claiming those integrations are verified.

The local TCC source remains the agreed temporary dependency. A public-clone
rehearsal must wait for a fetchable revision; this cleanup does not resolve it.

Cleanup verification: the core build succeeded; all 73 C++/CUDA tests and 36
Python tests passed, with the same four Python skips. The rebuilt executable
also passed the independent CODIF replay comparison. Logs are saved as
`build/joss-evidence/starspin-tidy-*.log`; replay artifacts are in
`build_joss_clean/tidy-replay/`. No live observing was performed.

Arrival-mask fix verification: both new regressions failed against the previous
code and passed after the ownership change. All 58 tests across the harness,
adaptive, correlate-and-beam, Starspin, fine-channelized Starspin, and writer
suites passed. The adaptive fixture-generator case skips standalone execution;
its separate `AdaptiveDetectionParityPythonCheck` passed and exercised it.
Before/after build and test logs are in `build/joss-evidence/starspin-arrivals-*.log`.

## External configuration and synthetic-fixture validation

Deployment JSON, compiled-in antenna wiring, site launch defaults, operational
records and site-specific notebooks/helpers were preserved in an external local
archive and removed from the public working tree. Paper attribution and required
notices remain. See `deployment.md`; Git history and ignored artifacts are outside
this working-tree cleanup.

The public fixture is generated deterministically by `generate_codif_fixture.py`.
Its new Starspin replay passed an independent comparison of 31,458,240 complex
visibilities (maximum relative RMS 9.36e-6). Packet/reassembly/config tests passed
51 cases; Python passed 39 with four optional skips. The isolated two-process
kernel UDP test received and processed 30,000 packets at about 5,000 packets/s,
with zero reported missing/discarded packets or NIC drops and four finite,
nonzero visibility integrations. This establishes a local socket smoke test,
not physical-NIC throughput or astronomical calibration.

## Reviewer workflow rerun (27 October 2026)

The maintainer confirmed the repository has been public throughout its development
history. The TCC upstream work is shelved; this run retained local TCC revision
`36e5edc3bc49772670edb5451f7b397342d26602` without adding John's proposed patch.

A frozen snapshot of the current uncommitted candidate was built in a new build
directory. Existing fetched dependency sources and the prepared CUDA image were
reused, but compiled objects were not. Unrelated untracked CryoPAF files and
`tests/test_dada_site_metadata.py` were excluded explicitly. The source archive
SHA-256 is `cbb3fccc47d70dfc8279ba1f6767f3e5a9dcd101b1dbfe5fa6591967a85e98d9`.
This identifies the tested source; it is not a published release commit.

- All **75 core C++/CUDA tests passed**: Starspin 6, fine-channelized Starspin 11,
  CPU packet/stream-map/CODIF tests 28, processor tests 21, writers 6, run audit 3.
- Python 3.10 with freshly built, pinned DiFX: **49 passed, 4 skipped**. The skips
  require the optional commissioning package (three) or CUDA delay-test library
  (one). A fresh Python 3.12 container installed the declared test requirements:
  **48 passed, 5 skipped**, with the additional skip being the external DiFX test
  already passed under Python 3.10. CFF 1.2.0 validation passed there too.
- Fine-channelized replay: **28,702,080 complex visibilities**, maximum relative
  RMS `9.355201319730967e-6`, maximum autopower-normalized error
  `5.46492831015539e-5`; comparison passed and both negative controls were rejected.
- Coarse DiFX comparison: **11 complete integrations / 19,360 complex
  visibilities**, maximum relative RMS `2.441710441305532e-8`, maximum
  autopower-normalized error `1.0594101263793658e-7`. All comparisons passed;
  conjugation, missing-scale and polarization-swap controls were rejected.
- No telescope, FPGA or live observing state was accessed.

Evidence: `build/joss-validation-20261027/summary.json` and
`reviewer-evidence.tar.gz` in that directory. The bundle includes source and
dependency archives, file hashes, source/submodule provenance, container and
package identities, commands, per-suite logs/XML, and replay data/reports.
Absolute paths in the logs are container paths under `/evidence`.

**Still open:** commit/push the scoped candidate and workflows, obtain passing
hosted CI, resolve the fetchable TCC dependency and repeat a clean recursive clone,
and have a colleague independently follow the reviewer guide. GitHub reported
zero workflow runs for `feat/difx-correlation-validation` when checked. Local
Python 3.12 checks do not constitute hosted CI or a colleague rehearsal.
