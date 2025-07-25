


## Building and Running Tests

On OzStar
```
module load cuda/12.6.0
module load gcc/13.3.0
module load cmake/3.29.3
```

To Build:
```
cd build
cmake ..
cmake --build .
```

To Test:
```
cd build
cmake .. -DBUILD_TESTING=ON
cmake --build .
ctest
```


## Next Steps
- [x] Understand the relationship between antenna & baseline in tcc.
- [x] Write code to get data from PCAP / CODIF / DADA data into right format for correlator & beamformer.


## TCC Tips

- The number of data points per block is fixed at 128 bytes * COMPLEX / (N_BITS_PER_COMPONENT * COMPLEX) = 128 / N_BITS_PER_COMPONENT. For instance for 8-bit components (8 x 2 = 16 bits complex) then there are 128/8 = 16 time points per block. COMPLEX = 1 if real or 2 if complex.
- The number of receivers must be a multiple of 32. If there are fewer than 32 receivers then set NR_RECEIVERS to 32 and zero pad. 
- NR_POLARIZATIONS must be 2 - if not then zero pad.

## CCGLIB Tips

- The second matrix in the calculation must be in column-major format only for the rows/columns sections.
  The Batch / Complex sections are not column-major.

## cuTENSOR Tips
- cuTENSOR assumes that given modes are column-major. To deal with this in tensor.cpp I introduce a std::reverse
  so that we can think in row-major geometries.

## TODOs

- [ ] Add multiple processing streams to de-couple processing of correlation matrix & transpositions.
- [x] Add integration tests around the current engine to make sure it doesn't get broken
- [x] Get rid of some of the original test code
- [ ] Take h_scales into account with the data.
- [x] Verify the visibilities from the test PCAP files.
- [x] Clean up the read_pcap file to move PCAP code out of main code file.
- [x] Do beamforming using ccglib.
- [ ] Figure out how to do this capturing packets.
- [x] Output formats for beams / debug information.
- [ ] Integrating correlations on the GPU over time and then dumping out to disk.
- [x] Profile and check that things are running concurrently.
- [ ] Decompose correlation matrix from lower-triangular form.
- [ ] Begin writing spatial filtering algorithms.
- [ ] Get benchmarks of initial work / slowest sections.
