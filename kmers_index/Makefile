buildIndex: buildIndex.cpp hashKMer.hpp loadKMerIndex.cpp hashKMerIndex.h
	g++ buildIndex.cpp -o buildIndex -L.. -lbwa -lm -lz -lpthread

.PHONY: run
run:
	./buildIndex ../../gaivi_test/HG38/GCF_000001405.39_GRCh38.p13_genomic.fna
